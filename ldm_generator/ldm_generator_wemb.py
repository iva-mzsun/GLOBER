import einops
import os
import torch
import torch as th
import torch.nn as nn
from omegaconf import OmegaConf

from tqdm import tqdm
from einops import rearrange, repeat
from ldm_generator.diffusion.ddim import DDIMSampler
from ldm_generator.diffusion.ddpm import LatentDiffusion
from utils.util import instantiate_from_config

from ipdb import set_trace as st

class Generator_LDM(LatentDiffusion):
    def __init__(self,
                 stable_ae_config,
                 videocontent_key,
                 vc_size, vc_channels,
                 latent_size, latent_channels,
                 optimizer="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vc_size = vc_size
        self.vc_channels = vc_channels
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.optimizer = optimizer
        self.videocontent_key = videocontent_key

        # Dataset Condition: Initialize learnable embedding for current dataset
        self.context_emb = nn.Parameter(torch.randn(77, 768) * 0.1)

        # video auto encoder
        self.stable_ae = instantiate_from_config(stable_ae_config)
        for p in self.stable_ae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # _, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs) # x: [b c h w], c: [b c]
        # encode full video frames
        full_frames = batch[self.videocontent_key].to(self.device)
        if self.use_fp16:
            full_frames = full_frames.to(memory_format=torch.contiguous_format).half()
        else:
            full_frames = full_frames.to(memory_format=torch.contiguous_format).float()
        T = full_frames.shape[2]
        full_frames = einops.rearrange(full_frames, 'b c t h w -> (b t) c h w')
        vidae_input = self.encode_first_stage(full_frames)
        vidae_input = self.get_first_stage_encoding(vidae_input).detach()
        vidae_input = rearrange(vidae_input, '(b t) c h w -> b c t h w', t=T)
        videocontent = self.stable_ae.encode(vidae_input).detach()

        c = repeat(self.context_emb, 'l c -> b l c',
                   b=videocontent.shape[0])

        if bs is not None:
            videocontent, c = videocontent[:bs], c[:bs]

        return videocontent, dict(c_crossattn=[c])

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def decode_vc_parrallel(self, vidcontent, uc_vidcontent, n_frames, verbose=False):
        frame_sample = self.stable_ae.decode(vidcontent, uc_vidcontent, n_frames)
        frame = self.decode_first_stage(frame_sample)  # (b t), c, h, w
        cur_video = rearrange(frame, '(b t) c h w -> b c t h w', t=n_frames)
        return cur_video

    @torch.no_grad()
    def log_videos(self, batch, N=8, x_T=None, n_frames=4, sample=True,
                   use_ddim=True, ddim_steps=50, ddim_eta=0.0,
                   verbose=False, unconditional_guidance_scale=9.0, **kwargs):
        log = dict()
        N = min(batch[self.videocontent_key].shape[0], N)
        # obtain inputs & conditions
        assert use_ddim == (ddim_steps > 0)
        # obtain conditions
        context = repeat(self.context_emb, 'l c -> b l c', b=N)
        uc_context = torch.zeros_like(context)
        new_cond = {"c_crossattn": [context]}
        uc_cond = {"c_crossattn": [uc_context]}
        ddim_kwargs = {"use_ddim": use_ddim,
                       "ddim_eta": ddim_eta,
                       "ddim_steps": ddim_steps}
        # decode video content & visualize input full video frames
        full_frames = batch[self.videocontent_key][:N].to(self.device)
        log['full_frames'] = full_frames.to(self.device) # b, c, t, h, w

        T = full_frames.shape[2]
        uc_full_frames = torch.zeros_like(full_frames)
        uc_full_frames = einops.rearrange(uc_full_frames,
                                          'b c t h w -> (b t) c h w')
        uc_full_frames = self.encode_first_stage(uc_full_frames)
        uc_full_frames = self.get_first_stage_encoding(uc_full_frames).detach()
        uc_full_frames = rearrange(uc_full_frames, '(b t) c h w -> b c t h w', t=T)
        uc_global_feat = self.stable_ae.encode(uc_full_frames).detach()

        full_frames = einops.rearrange(full_frames,
                                       'b c t h w -> (b t) c h w')
        full_frames = self.encode_first_stage(full_frames)
        full_frames = self.get_first_stage_encoding(full_frames).detach()
        full_frames = rearrange(full_frames, '(b t) c h w -> b c t h w', t=T)
        gt_global_feat = self.stable_ae.encode(full_frames).detach()
        gt_samples = self.decode_vc_parrallel(gt_global_feat, uc_global_feat, n_frames)
        log['samples_rec'] = gt_samples

        if sample:
            global_feat, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                           verbose=verbose, **ddim_kwargs)
            torch.cuda.empty_cache()
            samples = self.decode_vc_parrallel(global_feat, uc_global_feat, n_frames)
            log["samples"] = samples
        # sampling with unconditional guidance scale
        if unconditional_guidance_scale >= 0.0:
            global_feat, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                           unconditional_conditioning=uc_cond,
                                           unconditional_guidance_scale=unconditional_guidance_scale,
                                           verbose=verbose, **ddim_kwargs)
            torch.cuda.empty_cache()
            samples = self.decode_vc_parrallel(global_feat, uc_global_feat, n_frames)
            log[f"samples_ug_scale_{unconditional_guidance_scale:.2f}"] = samples
        torch.cuda.empty_cache()
        return log


    @torch.no_grad()
    def sample_log(self, cond, batch_size, use_ddim, ddim_steps, x_T, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (batch_size, self.vc_channels, self.vc_size, self.vc_size)
        noise = torch.randn(shape, device=self.device)
        x_T = noise if x_T is None else noise
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     cond, x_T=x_T, verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        params += [self.context_emb]

        # == count total params to optimize ==
        optimize_params = 0
        for param in params:
            optimize_params += param.numel()
        print(f"NOTE!!! {optimize_params/1e6:.3f}M params to optimize in TOTAL!!!")

        # == load optimizer ==
        if self.optimizer.lower() == "adam":
            print("Load AdamW optimizer !!!")
            opt = torch.optim.AdamW(params, lr=lr)
        else:
            raise NotImplementedError

        # == load lr scheduler ==
        if self.use_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
