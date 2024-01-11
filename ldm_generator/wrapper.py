import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict
from utils.util import instantiate_from_config
from ldm_autoencoder.diffusion.ddim_v1 import DDIMSampler
from ldm_generator.diffusion.ddpm import DDPM
from ipdb import set_trace as st
from einops import repeat, rearrange

class StableAutoencoder(DDPM):
    def __init__(self, enc_cfg, ckpt_path,
                 decode_sample_cfg,
                 *args, **kwargs):
        super(StableAutoencoder, self).__init__(*args, **kwargs)
        enc_cfg['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        enc_cfg['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.enc = instantiate_from_config(enc_cfg)
        self.decode_sample_cfg = decode_sample_cfg
        # initialize learnable context
        self.context_emb = nn.Parameter(torch.randn(77, 768) * 0.1)
        # Video Condition: Projecting global feature
        self.global_projector = nn.Sequential(
            nn.Conv2d(self.enc.out_channels, 768,
                      kernel_size=7, padding=3, stride=2),
            nn.SiLU(),
            nn.Conv2d(768, 768, kernel_size=3, padding=1)
        )
        # if ckpt_path:
        self.ckpt_path= ckpt_path
        self.load_ckpt(ckpt_path)
        for p in self.parameters():
            p.require_grad = False
        self.shape = None

    def load_ckpt(self, ckpt):
        state = torch.load(ckpt, map_location="cpu")
        # torch.load(self.ckpt_path, map_location="cpu")['state_dict']
        state_dict = state['state_dict']
        enc_state_dict = OrderedDict()
        dec_state_dict = OrderedDict()
        proj_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'global_projector' in k:
                newk = k.replace('global_projector.', '')
                proj_state_dict[newk] = v
            elif 'videocontent_enc' in k:
                newk = k.replace('videocontent_enc.', '')
                enc_state_dict[newk] = v
            elif 'model.diffusion_model.' in k:
                newk = k.replace('model.diffusion_model.',
                                 'diffusion_model.')
                dec_state_dict[newk] = v
            elif 'context_emb' in k:
                context_emb_values = v
            # else:
            #     print(f"Unused keys for auto-encoder: {k}")
        print(f'Loaded StableAE state_dict from [{ckpt}]')
        self.context_emb.data = context_emb_values
        self.enc.load_state_dict(enc_state_dict, strict=True)
        self.model.load_state_dict(dec_state_dict, strict=True)
        self.global_projector.load_state_dict(proj_state_dict, strict=True)
        for p in self.parameters():
            p.require_grad = False

    @torch.no_grad()
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_frame_pre = cond['noisy_frame_pre']
        # dataset-specific input
        context_emb = cond['prompt']
        # obtain global features outputted by the video encoder
        if cond.get('frame_feats', None) is not None:
            frame_feats = cond['frame_feats']
            global_feat, kl_loss = cond['global_feat'], None
        elif cond.get('global_feat', None) is None:  # during training or testing
            full_indexes = cond['full_index']
            key_frames, tar_indexes = cond['key_frames'], cond['tar_index']
            global_feat, kl_loss = self.videocontent_enc.encode(key_frames)
            frame_feats = self.videocontent_enc.decode(global_feat,
                                                       full_indexes, tar_indexes)
        else:
            tar_indexes = cond['tar_index']
            full_indexes = cond['full_index']
            global_feat, kl_loss = cond['global_feat'], None
            frame_feats = self.videocontent_enc.decode(global_feat,
                                                       full_indexes, tar_indexes)
        # video-specific feature
        nframes = context_emb.shape[0] // global_feat.shape[0]
        video_emb = self.global_projector(global_feat)
        video_emb = repeat(video_emb, 'b c h w -> b t c h w', t=nframes)
        video_emb = rearrange(video_emb, 'b t c h w -> (b t) (h w) c')
        # video_emb = rearrange(video_emb, 'b c h w -> b (h w) c')

        context = context_emb + video_emb
        eps = diffusion_model(x=x_noisy, context=context,
                              vc_feats=frame_feats, timesteps=t,
                              preframe_noisy=cond_frame_pre)
        return eps

    @torch.no_grad()
    def encode(self, x):
        '''
            x: b, c, t, h, w
        '''
        if self.shape is None:
            self.shape = list(x.shape[1:])
        global_feat, _ = self.enc.encode(x, is_training=False)
        return global_feat # b c h w

    @torch.no_grad()
    def decode(self, global_feat, uc_global_feat, n_frames=4, video_length=16, verbose=False):
        '''
            vc: b, c, h, w
            index: b, t
            context: b, l, c
        '''
        N = global_feat.shape[0]
        # obtain context embeddings
        c = repeat(self.context_emb, 'l c -> b l c', b=N)
        # indexes of target video frames
        tar_index = [torch.ones((N, 1)) * t for t in range(n_frames)]
        tar_index = torch.cat(tar_index, dim=1).to(self.device) / n_frames  # b t
        full_index = [torch.ones((N, 1)) * t for t in range(video_length)]
        full_index = torch.cat(full_index, dim=1).to(self.device) / video_length  # b t
        # obtain conditional/unconditional inputs: textual descriptions
        c = repeat(c, 'b c l -> b t c l', t=n_frames)
        c = rearrange(c, 'b t c l -> (b t) c l')
        uc = torch.zeros_like(c)

        # obtain conditional/unconditional inputs: vc feats
        frame_feats = self.enc.decode(global_feat, full_index, tar_index)
        uc_frame_feats = self.enc.decode(uc_global_feat, full_index, tar_index)
        # collect conditional/unconditional inputs
        c_cond = dict(prompt=c, frame_feats=frame_feats, global_feat=global_feat)
        uc_cond_video = dict(prompt=c, frame_feats=uc_frame_feats, global_feat=uc_global_feat)
        uc_cond_domain = dict(prompt=uc, frame_feats=uc_frame_feats, global_feat=uc_global_feat)

        # sampling
        ddim_sampler = DDIMSampler(self)
        use_ddim = self.decode_sample_cfg.use_ddim
        ddim_eta = self.decode_sample_cfg.ddim_eta
        ddim_steps = self.decode_sample_cfg.ddim_steps
        ucgs_video = self.decode_sample_cfg.ucgs_video
        ucgs_domain = self.decode_sample_cfg.ucgs_domain
        unconditional_guidance_scale = dict(domain=ucgs_domain, video=ucgs_video)
        unconditional_conditioning = dict(domain=uc_cond_domain, video=uc_cond_video)

        batch_size = N * n_frames
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                         n_frames, c_cond, verbose=verbose,
                                         ddim=use_ddim, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
                                         unconditional_conditioning=unconditional_conditioning,
                                         unconditional_guidance_scale=unconditional_guidance_scale)

        return samples





