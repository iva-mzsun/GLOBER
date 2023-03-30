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
from ldm.util import instantiate_from_config

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

        # video auto encoder
        self.stable_ae = instantiate_from_config(stable_ae_config)
        for p in self.stable_ae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        _, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs) # x: [b c h w], c: [b c]
        # encode full video frames
        vidcontent = batch[self.videocontent_key].to(self.device)
        if self.use_fp16:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).half()
        else:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).float()
        T = vidcontent.shape[2]
        vidcontent = rearrange(vidcontent, 'b c t h w -> (b t) c h w')
        vidcontent = self.encode_first_stage(vidcontent)
        vidcontent = self.get_first_stage_encoding(vidcontent).detach()
        vidcontent = rearrange(vidcontent, '(b t) c h w -> b c t h w', t=T)
        vc = self.stable_ae.encode(vidcontent).detach()
        if bs is not None:
            vc, c = vc[:bs], c[:bs]
        return vc, dict(c_crossattn=[c])

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def decode_vc(self, vc, video_length, context, uc_context, verbose=False):
        video_sample = []
        # for index in tqdm(range(video_length), desc=f"Decoding video frames..."):
        for index in range(video_length):
            ind = torch.ones(vc.shape[0]).unsqueeze(1).to(vc.dtype).to(vc.device) * index / video_length
            frame_sample = self.stable_ae.decode(vc, index=ind, context=context,
                                                 uc_context=uc_context, verbose=verbose)
            frame = self.decode_first_stage(frame_sample) # b, c, h, w
            frame = frame.unsqueeze(2)
            video_sample.append(frame)
        cur_video = torch.cat(video_sample, dim=2) # b, c, t, h, w
        return cur_video

    @torch.no_grad()
    def log_videos(self, batch, N=8, x_T=None, n_frames=4, sample=True, ddim_steps=50, ddim_eta=0.0,
                   verbose=False, unconditional_guidance_scale=3.0, **kwargs):
        N = min(batch[self.videocontent_key].shape[0], N)
        # obtain inputs & conditions
        use_ddim = ddim_steps is not None
        vc, cond = self.get_input(batch, self.first_stage_key, bs=N)
        # obtain conditions
        context = cond["c_crossattn"][0]
        uc_context = self.get_unconditional_conditioning(N)
        # shared kwargs
        ddim_kwargs = {"use_ddim": use_ddim, "ddim_steps": ddim_steps, "ddim_eta": ddim_eta}
        decode_vc_kwargs = {"video_length": n_frames, "context": context, "uc_context": uc_context}
        # decode video content & visualize input full video frames
        log = dict()
        log["reconstruction"] = self.decode_vc(vc, **decode_vc_kwargs)
        log['full_frames'] = batch[self.videocontent_key].to(self.device) # b, c, t, h, w
        # full conditional sampling
        new_cond = {"c_crossattn": [context]}
        uc_cond = {"c_crossattn": [uc_context]}
        if sample or unconditional_guidance_scale <= 0.0:
            vc_sample, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                           verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc(vc_sample, **decode_vc_kwargs)
            log["samples"] = samples
        # sampling with unconditional guidance scale
        if unconditional_guidance_scale > 0.0:
            vc_sample, _ = self.sample_log(cond=new_cond, batch_size=N, x_T=x_T,
                                           unconditional_conditioning=uc_cond,
                                           unconditional_guidance_scale=unconditional_guidance_scale,
                                           verbose=verbose, **ddim_kwargs)
            samples = self.decode_vc(vc_sample, **decode_vc_kwargs)
            log[f"samples_ug_scale_{unconditional_guidance_scale:.2f}"] = samples
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
