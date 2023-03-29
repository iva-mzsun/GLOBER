import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm_generator.diffusion.ddpm import DDPM
from ipdb import set_trace as st

class StableAutoencoder(DDPM):
    def __init__(self, enc_cfg, ckpt_path,
                 decode_sample_cfg,
                 *args, **kwargs):
        super(StableAutoencoder, self).__init__(*args, **kwargs)
        enc_cfg['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        enc_cfg['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.enc = instantiate_from_config(enc_cfg)
        self.decode_sample_cfg = decode_sample_cfg
        self.ckpt_path= ckpt_path
        self.load_ckpt(ckpt_path)
        for p in self.parameters():
            p.require_grad = False

    def load_ckpt(self, ckpt):
        state = torch.load(ckpt, map_location="cpu")
        # torch.load(self.ckpt_path, map_location="cpu")['state_dict']
        state_dict = state['state_dict']

        enc_state_dict = OrderedDict()
        dec_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'videocontent_enc' in k:
                newk = k.replace('videocontent_enc.', '')
                enc_state_dict[newk] = v
            if 'model.diffusion_model.' in k:
                newk = k.replace('model.diffusion_model.',
                                 'diffusion_model.')
                dec_state_dict[newk] = v
        print(f'Loaded StableAE state_dict from [{ckpt}]')
        self.enc.load_state_dict(enc_state_dict, strict=True)
        self.model.load_state_dict(dec_state_dict, strict=True)
        self.shape = None

    @torch.no_grad()
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # obtain vc_feats
        vc = cond['c_vc'][0]
        index = cond['c_index'][0]
        # vc_feats = self.enc.get_feats(vc, index)
        vc_feats = self.enc.decode(vc, index)
        # obtain context
        cond_txt = torch.cat(cond['c_crossattn'], 1) # b, l, c
        eps = diffusion_model(x=x_noisy, vc_feats=vc_feats, timesteps=t, context=cond_txt)
        return eps

    @torch.no_grad()
    def encode(self, x):
        '''
            x: b, c, t, h, w
        '''
        if self.shape is None:
            self.shape = list(x.shape[1:])
        # vc = self.enc.get_latents(x)
        vc_posterior = self.enc.encode(x)
        vc = vc_posterior.mode()
        return vc # b c h w

    @torch.no_grad()
    # def decode(self, vc, index, context, uc_context, shape, batch_size, use_ddim,
    #            unconditional_guidance_scale, ddim_steps, ddim_eta, verbose=False):
    def decode(self, vc, index, context, uc_context, verbose=False):
        '''
            vc: b, c, h, w
            index: b, t
            context: b, l, c
        '''
        # condition
        # vc_feats = self.enc.get_feats(vc, index)
        cond = {"c_vc": [vc], "c_index": [index], "c_crossattn": [context]}
        # unconditional conditioning
        uc_x = torch.zeros([vc.shape[0]] + self.shape).to(vc.dtype)
        uc_vc = self.encode(uc_x.to(vc.device))
        uc_cond = {"c_vc": [uc_vc], "c_index": [index], "c_crossattn": [uc_context]}
        # sampling
        ddim_sampler = DDIMSampler(self)
        use_ddim = self.decode_sample_cfg.use_ddim
        ddim_eta = self.decode_sample_cfg.ddim_eta
        ddim_steps = self.decode_sample_cfg.ddim_steps
        ucgs = self.decode_sample_cfg.unconditional_guidance_scale
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ = ddim_sampler.sample(batch_size=vc.shape[0],
                                         S=ddim_steps,
                                         unconditional_conditioning=uc_cond,
                                         shape=shape, conditioning=cond, verbose=verbose,
                                         ddim=use_ddim, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
                                         unconditional_guidance_scale=ucgs)
        return samples





