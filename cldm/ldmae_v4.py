import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module, timestep_embedding
)

from tqdm import tqdm
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, \
    Downsample, normalization, ResBlock2n
from cldm.diffusion.ddpm_v4 import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.utils.blocks import ResBlockwoEmb, TemporalAttentionBlock, SpatialAttentionBlock, \
    SpatioTemporalAttentionBlock

from ipdb import set_trace as st

class SeqDiscriminator(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        legacy=True,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # create learnable embeddings to distinguish pre/cur video frame
        shape = (1, in_channels, 2, image_size, image_size)
        self.frame_embed = nn.Parameter(torch.randn(shape), requires_grad=True)

        # create input blocks
        self.input_blocks = nn.ModuleList([
             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        ])
        input_block_chans = [model_channels]
        ds, ch = 1, model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockwoEmb(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels

                    layers.append(
                        SpatioTemporalAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlockwoEmb(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channels=out_ch, dims=dims,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # obtain video content feature
        self.out = nn.Sequential(
            conv_nd(dims, ch, model_channels, 3, padding=1),
            normalization(model_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(), # B, C
            nn.Linear(model_channels, 1)
        )

    def forward(self, x, **kwargs):
        '''
        x: [(b t), c, 2, h, w]
        '''
        h = x.type(self.dtype)
        h = h + self.frame_embed
        for module in self.input_blocks:
            h = module(h)
        logit = self.out(h).squeeze(1)
        return logit # B

class VideoContentEnc(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        learnable_content,
        sdinput_block_ds,
        sdinput_block_chans,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        legacy=True,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.learnable_content = learnable_content

        # create learnable embedding
        if self.learnable_content:
            shape = (in_channels, image_size, image_size)
            self.video_content = nn.Parameter(torch.randn(shape), requires_grad=True)
        # create input blocks
        self.input_blocks = nn.ModuleList([
             conv_nd(dims, in_channels, model_channels, 3, padding=1)
        ])
        input_block_chans = [model_channels]
        ds, ch = 1, model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockwoEmb(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels

                    layers.append(
                        SpatialAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    layers.append(
                        TemporalAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlockwoEmb(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channels=out_ch, dims=dims,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # obtain video content feature
        if self.learnable_content:
            self.out_vc = nn.Sequential(
                conv_nd(dims, ch, self.out_channels, 3, padding=1),
                normalization(self.out_channels),
                nn.SiLU(),
                conv_nd(dims, self.out_channels, self.out_channels, 1)
            )
        else:
            self.out_vc = nn.Sequential(
                conv_nd(dims, ch, self.out_channels, 3, padding=1),
                normalization(self.out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, image_size // ds, image_size // ds)),
                conv_nd(dims, self.out_channels, self.out_channels, 1)
            )

        # embed index embed
        index_embed_dim = model_channels * 4
        self.index_embed = nn.Sequential(
            linear(model_channels, index_embed_dim),
            nn.SiLU(),
            linear(index_embed_dim, index_embed_dim),
        )

        # obtain output blocks
        self.output_blocks = nn.ModuleList([])
        for i, ch in enumerate(sdinput_block_chans):
            curds, n = ds, 0
            tards = sdinput_block_ds[i]
            if tards == ds:
                layers = [ResBlock(self.out_channels, index_embed_dim, dropout, ch)]
            elif ds < tards:
                while curds < tards:
                    n += 1
                    curds *= 2
                layers = [ResBlock2n(self.out_channels, index_embed_dim,
                                     dropout, ch, down=True, sample_num=n)]
            else:
                while curds > tards:
                    n += 1
                    curds /= 2
                layers = [ResBlock2n(self.out_channels, index_embed_dim,
                                     dropout, ch, up=True, sample_num=n)]
            layers.append(self.make_zero_conv(ch, dims=2))
            self.output_blocks.append(TimestepEmbedSequential(*layers))

            self.sdinput_block_ds = sdinput_block_ds
            self.sdinput_block_chans = sdinput_block_chans

    def make_zero_conv(self, channels, dims=None):
        dims = dims or self.dims
        return nn.Sequential(zero_module(conv_nd(dims, channels, channels, 1, padding=0)))

    def forward(self, x, index, **kwargs):
        '''
        x: [b, c, t, h, w]
        ind: [b, t]
        '''
        # obtain vc
        if self.learnable_content:
            B, C, T, H, W = x.shape
            vc = repeat(self.video_content, 'c h w -> b c t h w', b=B, t=1)
            x = torch.cat([vc, x], dim=2)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)

        if self.learnable_content:
            h = self.out_vc(h)
            vc = h[:, :, 0, :, :] # B, C, H, W
        else:
            vc = self.out_vc(h) # B, C, 1, H, W
            vc = vc.squeeze(2) # B, C, H, W

        # obtain index embedding
        index = rearrange(index, 'b t -> (b t)')
        index_emb = timestep_embedding(index, self.model_channels)
        index_emb = self.index_embed(index_emb)
        # obtain insert feat
        vc_feats = []
        for module in self.output_blocks:
            vc_feat = module(vc, index_emb)
            vc_feats.append(vc_feat)

        return vc_feats

class AEUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, vc_feats=None, **kwargs):
        # obtain timestep embeddings
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
        # encode: insert vc feats
        hs = []
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            h = h + vc_feats[i]
            hs.append(h)
        # decode
        h = self.middle_block(h, emb, context)
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)

class AutoEncLDM(LatentDiffusion):

    def __init__(self,
                 generator_loss_weight,
                 seq_discriminator_config,
                 adversarial_loss,
                 videoenc_config, videocontent_key,
                 frame_index_key, optimize_params_key,
                 sd_lock_output, sd_lock_input, sd_lock_middle,
                 xrec_label_real=True,
                 optimizer="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.sd_lock_input = sd_lock_input
        self.sd_lock_middle = sd_lock_middle
        self.sd_lock_output = sd_lock_output
        self.optimize_params_key = optimize_params_key
        self.xrec_label_real = xrec_label_real
        # obtain video content
        self.videocontent_key = videocontent_key
        videoenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        videoenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.videocontent_enc = instantiate_from_config(videoenc_config)
        # obtain target video frame indexes
        self.frame_index_key = frame_index_key
        # obtain discriminator
        self.seq_discriminator = instantiate_from_config(seq_discriminator_config)
        self.generator_loss_weight = generator_loss_weight
        if adversarial_loss == "hinge":
            from cldm.utils.gan_loss import hinge_d_loss
            self.adversarial_loss = hinge_d_loss
        else:
            raise NotImplementedError

    def obtain_disc_label(self, batch, x, real):
        # label: 0 - fake, 1 - real
        B, T = batch[self.frame_index_key].shape

        cur_indexes = batch[self.frame_index_key] # [b, t]
        pre_indexes = torch.roll(cur_indexes, shifts=1, dims=1)
        cur_indexes = rearrange(cur_indexes, 'b t -> (b t)')
        pre_indexes = rearrange(pre_indexes, 'b t -> (b t)')
        if real:
            label = pre_indexes < cur_indexes
        else:
            label = torch.zeros_like(pre_indexes)

        curx = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        prex = torch.roll(curx, shifts=1, dims=1)
        curx = rearrange(curx, 'b t c h w -> (b t) c h w')
        prex = rearrange(prex, 'b t c h w -> (b t) c h w')
        curx = curx.unsqueeze(2)
        prex = prex.unsqueeze(2)
        outx = torch.cat([prex, curx], dim=2) # (b t) c 2 h w
        return outx, label

    def training_step(self, batch, batch_idx, optimizer_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        if optimizer_idx == 0:
            loss, loss_dict, _, x_recon = self.shared_step(batch, returnx=True) # x: [(b t) c h w]
            # log lr and custom losses
            lr = self.optimizers()[optimizer_idx].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            # log discriminator loss
            cat_x, label = self.obtain_disc_label(batch, x_recon, real=True)
            for p in self.seq_discriminator.parameters():
                p.requires_grad = True
            logit = self.seq_discriminator(cat_x)
            g_loss = self.adversarial_loss(logit, label) # reverse label
            self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss + g_loss * self.generator_loss_weight

        if optimizer_idx == 1:
            with torch.no_grad():
                _, _, x_start, x_recon = self.shared_step(batch, returnx=True)  # x: [(b t) c h w]
                x_start, x_recon = x_start.detach(), x_recon.detach()
            cat_x_start, label_start = self.obtain_disc_label(batch, x_start, real=True)
            cat_x_recon, label_recon = self.obtain_disc_label(batch, x_recon,
                                                              real=self.xrec_label_real)
            logit_start = self.seq_discriminator(cat_x_start)
            logit_recon = self.seq_discriminator(cat_x_recon)
            d_loss_start = self.adversarial_loss(logit_start, label_start)
            d_loss_recon = self.adversarial_loss(logit_recon, label_recon)
            d_loss = (d_loss_start + d_loss_recon) / 2
            lr = self.optimizers()[optimizer_idx].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/d_loss_start", d_loss_start, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/d_loss_recon", d_loss_recon, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return d_loss

    def p_losses(self, x_start, cond, t, returnx=False, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_noisy, t=t,
                                                    noise=model_output)
        elif self.parameterization == "x0":
            x_recon = model_output
        else:
            raise NotImplementedError()

        if returnx:
            return loss, loss_dict, x_start, x_recon
        return loss, loss_dict

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, repeat_c_by_T=True, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key,
                                 repeat_c_by_T=repeat_c_by_T,
                                 bs=bs, *args, **kwargs) # x: [b c h w], c: [b c]
        # encode full video frames
        vidcontent = batch[self.videocontent_key].to(self.device)
        if self.use_fp16:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).half()
        else:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).float()
        T = vidcontent.shape[2]
        vidcontent = einops.rearrange(vidcontent, 'b c t h w -> (b t) c h w')
        vidcontent = self.encode_first_stage(vidcontent)
        vidcontent = self.get_first_stage_encoding(vidcontent).detach()
        vidcontent = rearrange(vidcontent, '(b t) c h w -> b c t h w', t=T)
        # obtain target frame index embedding
        frame_indexes = batch[self.frame_index_key].to(self.device)
        if bs is not None:
            vidcontent = vidcontent[:bs]
            frame_indexes = frame_indexes[:bs]
        return x, dict(c_crossattn=[c], c_video=[vidcontent], c_index=[frame_indexes])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1) # b, l, c
        cond_vc = torch.cat(cond['c_video'], 2) # b, c, t, h, w
        cond_index = cond['c_index'][0] # b
        # cond_index = torch.cat(cond['c_index'], 1) # b, 1

        vc_feats = self.videocontent_enc(x=cond_vc, index=cond_index)
        eps = diffusion_model(x=x_noisy, vc_feats=vc_feats, timesteps=t, context=cond_txt)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=16, sample=False, ddim_steps=50, ddim_eta=0.0,
                   plot_denoise_rows=False, verbose=False, unconditional_guidance_scale=9.0, **kwargs):
        # obtain inputs
        use_ddim = ddim_steps is not None
        z, cond = self.get_input(batch, self.first_stage_key, bs=N)
        # obtain conditions
        N = min(z.shape[0], N)
        vc = cond["c_video"][0][:N]
        ind = cond["c_index"][0][:N]
        c = cond["c_crossattn"][0][:N]
        new_cond = {"c_video": [vc], "c_index": [ind], "c_crossattn": [c]}
        # obtain logs
        log = dict()
        log["reconstruction"] = self.decode_first_stage(z[:N])
        log['full_frames'] = rearrange(batch[self.videocontent_key].to(self.device),
                                       'b c t h w -> (b t) c h w')
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        # start sampling

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=new_cond,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_video = torch.zeros_like(vc)
            uc_full = {"c_crossattn": [uc_cross], "c_video": [uc_video], "c_index": [ind]} #TODO
            samples_cfg, _ = self.sample_log(cond=new_cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             verbose=verbose)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def log_videos(self, batch, N=2, n_frames=4, sample=True, ddim_steps=50, ddim_eta=0.0,
                   verbose=False, unconditional_guidance_scale=9.0, **kwargs):
        B, T = batch[self.frame_index_key].shape
        N = min(B, N)
        # obtain inputs
        use_ddim = ddim_steps is not None
        z, cond = self.get_input(batch, self.first_stage_key,
                                 bs=N, repeat_c_by_T=False)
        td = tqdm(range(n_frames))
        # obtain conditions
        vc = cond["c_video"][0] # B, C, T, H, W
        ind = cond["c_index"][0][:, :1] # B, 1
        c = cond["c_crossattn"][0]
        new_cond = {"c_video": [vc], "c_crossattn": [c]}
        # obtain logs
        log = dict()
        log["reconstruction"] = self.decode_first_stage(z)
        log['full_frames'] = rearrange(batch[self.videocontent_key].to(self.device),
                                       'b c t h w -> b c h (t w)')
        # start sampling
        if sample:
            samples = []
            for f in td:
                ind = torch.ones_like(ind) * f / n_frames
                new_cond["c_index"] = [ind]
                frames, _ = self.sample_log(cond=new_cond, batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose)
                samples.append(self.decode_first_stage(frames).unsqueeze(1))
            x_samples = torch.cat(samples, dim=1) # b, t, c, h, w
            x_samples = rearrange(x_samples, 'b t c h w -> (b t) c h w')
            log["samples"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc_video = torch.zeros_like(vc)
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cond = {"c_crossattn": [uc_cross], "c_video": [uc_video]}
            samples_ug = []
            for f in td:
                ind = torch.ones_like(ind) * f / n_frames
                uc_cond["c_index"] = [ind]
                new_cond["c_index"] = [ind]
                frames_ug, _ = self.sample_log(cond=new_cond, batch_size=N, ddim=use_ddim,
                                               ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose,
                                               unconditional_conditioning=uc_cond,
                                               unconditional_guidance_scale=unconditional_guidance_scale)
                samples_ug.append(self.decode_first_stage(frames_ug).unsqueeze(1))
            x_samples_ug = torch.cat(samples_ug, dim=1)  # b, t, c, h, w
            x_samples_ug = rearrange(x_samples_ug, 'b t c h w -> b c h (t w)')
            log[f"samples_ug_scale_{unconditional_guidance_scale:.2f}"] = x_samples_ug
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     cond, verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        # == parameters of generator ==
        g_params = list(self.videocontent_enc.parameters())
        # add select params
        for n, p in self.model.diffusion_model.named_parameters():
            flag = False
            if self.optimize_params_key is not None:
                for key in self.optimize_params_key:
                    flag = True if key in n else flag
            flag = flag or (n[:5] == "input" and not self.sd_lock_input)
            flag = flag or (n[:3] == "out" and not self.sd_lock_output)
            flag = flag or (n[:6] == "middle" and not self.sd_lock_middle)
            if flag:
                print("Add optimize parameters: ", n)
                g_params.append(p)

        # == parameters of discriminator ==
        d_params = list(self.seq_discriminator.parameters())

        # == count total params to optimize ==
        optimize_params = 0
        for param in g_params + d_params:
            optimize_params += param.numel()
        print(f"NOTE!!! {optimize_params/1e6:.3f}M params to optimize in TOTAL!!!")

        if self.optimizer == "hybridadam":
            raise NotImplementedError
        else:
            print("Load AdamW optimizer.")
            opt_g = torch.optim.AdamW(g_params, lr=lr)
            opt_d = torch.optim.AdamW(d_params, lr=lr)

        # if self.use_scheduler:
        #     from torch.optim.lr_scheduler import LambdaLR
        #     assert 'target' in self.scheduler_config
        #     scheduler = instantiate_from_config(self.scheduler_config)
        #     print("Setting up LambdaLR scheduler...")
        #     scheduler_g = [{
        #             'scheduler': LambdaLR(opt_g, lr_lambda=scheduler.schedule),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]
        #     scheduler_d = [{
        #         'scheduler': LambdaLR(opt_d, lr_lambda=scheduler.schedule),
        #         'interval': 'step',
        #         'frequency': 1
        #     }]
        #     return [opt_g, opt_d], [scheduler_g, scheduler_d]
        return [opt_g, opt_d], []

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
