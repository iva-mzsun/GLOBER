import einops
import torch
import torch as th
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange, repeat

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import log_txt_as_img, instantiate_from_config, default
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.openaimodel import UNetModel, \
    TimestepEmbedSequential, ResBlock, Downsample, normalization, ResBlock2n
from ldm.modules.diffusionmodules.util import conv_nd, linear, \
    zero_module, timestep_embedding

from ldm_autoencoder.diffusion.ddpm_v1 import LatentDiffusion
from ldm_autoencoder.utils.blocks import ResBlockwoEmb, TemporalAttentionBlock, \
    SpatialAttentionBlock, SpatioTemporalAttentionBlock

from ipdb import set_trace as st

class Discriminator(nn.Module):
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
        temporal_embeddings=False
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

        # temporal embeddings
        # create learnable embeddings to distinguish pre/cur video frame
        self.temporal_embed = temporal_embeddings
        if temporal_embeddings:
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
        if self.temporal_embed:
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
        self.out_channels = out_channels * 2
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
                layers = [ResBlock(out_channels, index_embed_dim, dropout, ch)]
            elif ds < tards:
                while curds < tards:
                    n += 1
                    curds *= 2
                layers = [ResBlock2n(out_channels, index_embed_dim,
                                     dropout, ch, down=True, sample_num=n)]
            else:
                while curds > tards:
                    n += 1
                    curds /= 2
                layers = [ResBlock2n(out_channels, index_embed_dim,
                                     dropout, ch, up=True, sample_num=n)]
            layers.append(self.make_zero_conv(ch, dims=2))
            self.output_blocks.append(TimestepEmbedSequential(*layers))

            self.sdinput_block_ds = sdinput_block_ds
            self.sdinput_block_chans = sdinput_block_chans

    def make_zero_conv(self, channels, dims=None):
        dims = dims or self.dims
        return nn.Sequential(zero_module(conv_nd(dims, channels, channels, 1, padding=0)))

    def encode(self, x):
        if self.learnable_content:
            B, C, T, H, W = x.shape
            vc = repeat(self.video_content, 'c h w -> b c t h w', b=B, t=1)
            x = torch.cat([vc, x], dim=2)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
        if self.learnable_content:
            h = self.out_vc(h)
            vc_moments = h[:, :, 0, :, :]  # B, C, H, W
        else:
            vc_moments = self.out_vc(h)  # B, C, 1, H, W
            vc_moments = vc_moments.squeeze(2)  # B, C, H, W
        vc_posterior = DiagonalGaussianDistribution(vc_moments)
        return vc_posterior # [B, C * 2, H, W]

    def decode(self, vc, index):
        """
            vc: B, C, H, W
            index: B, T
        """
        # resize vc
        B, T = index.shape
        vc = repeat(vc, 'b c h w -> b c t h w', t=T)
        vc = rearrange(vc, 'b c t h w -> (b t) c h w')
        # obtain index embedding
        index = rearrange(index, 'b t -> (b t)').to(vc.device)
        index_emb = timestep_embedding(index, self.model_channels)
        index_emb = self.index_embed(index_emb)
        # obtain insert feat
        vc_feats = []
        for module in self.output_blocks:
            vc_feat = module(vc, index_emb)
            vc_feats.append(vc_feat)
        return vc_feats

    def forward(self, x, index, is_training=False, **kwargs):
        '''
            x: [b, c, t, h, w]
            ind: [b, t]
        '''
        vc_posterior = self.encode(x)
        if is_training is True:
            vc = vc_posterior.sample()
            kl_loss = vc_posterior.kl()
        else:
            vc = vc_posterior.mode()
            kl_loss = None
        vc_feats = self.decode(vc, index)
        return vc_feats, kl_loss

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
                 videoenc_config, videocontent_key,
                 frame_index_key, optimize_params_key,
                 sd_lock_output, sd_lock_input, sd_lock_middle,
                 allow_gan, gan_config, kl_loss_weight=1e-6,
                 optimizer="adam", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.sd_lock_input = sd_lock_input
        self.sd_lock_middle = sd_lock_middle
        self.sd_lock_output = sd_lock_output
        self.kl_loss_weight = kl_loss_weight
        self.optimize_params_key = optimize_params_key
        # obtain video content
        self.videocontent_key = videocontent_key
        videoenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        videoenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.videocontent_enc = instantiate_from_config(videoenc_config)
        # obtain target video frame indexes
        self.frame_index_key = frame_index_key
        # initialize gan configs
        self.allow_gan = allow_gan
        if allow_gan:
            assert gan_config is not None
            self.init_gan_configs(gan_config)

    def init_gan_configs(self, cfg):
        if cfg.adversarial_loss == "hinge":
            from ldm_autoencoder.utils.gan_loss import hinge_d_loss
            self.adversarial_loss = hinge_d_loss
        else:
            raise NotImplementedError
        self.generator_frequency = cfg.generator_frequency
        self.discriminator_frequency = cfg.discriminator_frequency
        self.generator_loss_weight = cfg.generator_loss_weight
        self.discriminator = instantiate_from_config(cfg.discriminator)

    def obtain_disc_label(self, batch, realx, fakex, optimizer_idx):
        ''' Ensure indexes of inputs are sorted!!!
        realx/fakex: [B * T, C, H, W]
        '''
        B, T = batch[self.frame_index_key].shape
        realx = rearrange(realx, '(b t) c h w -> b c t h w', b=B, t=T)
        fakex = rearrange(fakex, '(b t) c h w -> b c t h w', b=B, t=T)
        real0 = realx[:, :, :1, :, :] # B, C, 1, H, W
        fake0 = fakex[:, :, :1, :, :]  # B, C, 1, H, W
        real1 = realx[:, :, -1:, :, :]  # B, C, 1, H, W
        fake1 = fakex[:, :, -1:, :, :]  # B, C, 1, H, W

        samples, labels = [], []
        if optimizer_idx == 0:  # train generator
            # True samples:
            samples.append(torch.cat([real0, fake1], dim=2))
            labels.append(torch.ones(B).to(self.dtype))
            samples.append(torch.cat([fake0, real1], dim=2))
            labels.append(torch.ones(B).to(self.dtype))
            samples.append(torch.cat([fake0, fake1], dim=2))
            labels.append(torch.ones(B).to(self.dtype))
        elif optimizer_idx == 1: # train discriminator
            # TRUE samples:
            samples.append(torch.cat([real0, real1], dim=2))
            labels.append(torch.ones(B).to(self.dtype))
            # False samples:
            samples.append(torch.cat([real0, fake1], dim=2))
            labels.append(torch.zeros(B).to(self.dtype))
            samples.append(torch.cat([fake0, real1], dim=2))
            labels.append(torch.zeros(B).to(self.dtype))
            samples.append(torch.cat([fake0, fake1], dim=2))
            labels.append(torch.zeros(B).to(self.dtype))
            samples.append(torch.cat([real1, real0], dim=2))
            labels.append(torch.zeros(B).to(self.dtype))
            samples.append(torch.cat([fake1, fake0], dim=2))
            labels.append(torch.zeros(B).to(self.dtype))
        # concate all samples
        samples = torch.cat(samples, dim=0) # (b * 3 or 4) c 2 h w
        labels = torch.cat(labels, dim=0).to(samples.device) # (b * 3 or 4)
        return samples, labels

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.allow_gan:
            return self.training_step_wgan(batch, batch_idx, optimizer_idx)
        else:
            return self.training_step_wogan(batch, batch_idx)

    def training_step_wogan(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def training_step_wgan(self, batch, batch_idx, optimizer_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        if optimizer_idx == 0:
            loss, loss_dict, x_start, x_recon = self.shared_step(batch, is_training=True, returnx=True)
            # log lr and custom losses
            lr = self.optimizers()[optimizer_idx].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            # obtain temporal discriminator results
            for p in self.discriminator.parameters():
                p.requires_grad = True
            samplex, labels = self.obtain_disc_label(batch, x_start, x_recon, 0)
            logits = self.discriminator(samplex)
            g_loss = self.adversarial_loss(logits, labels)
            self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss + g_loss * self.generator_loss_weight

        if optimizer_idx == 1: # train discriminator
            with torch.no_grad():
                loss, _, x_start, x_recon = self.shared_step(batch, is_training=True, returnx=True)
                x_start, x_recon = x_start.detach(), x_recon.detach()
            # obtain spatial discriminator results
            samplex, labels = self.obtain_disc_label(batch, x_start, x_recon, 1)
            logits = self.discriminator(samplex)
            d_loss = self.adversarial_loss(logits, labels)
            # log
            lr = self.optimizers()[optimizer_idx].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return d_loss

    def p_losses(self, x_start, cond, t, noise=None, is_training=False, returnx=False):
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if is_training is True:
            model_output, kl_loss = self.apply_model(x_noisy, t, cond, is_training)
        else:
            kl_loss = None
            model_output = self.apply_model(x_noisy, t, cond, is_training)

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

        if kl_loss is not None:
            loss += kl_loss.mean() * self.kl_loss_weight
            loss_dict.update({f'{prefix}/kl_loss': kl_loss.mean()})

        if returnx:
            if self.parameterization == "eps":
                x_recon = self.predict_start_from_noise(x_noisy, t=t,
                                                        noise=model_output)
            elif self.parameterization == "x0":
                x_recon = model_output
            else:
                raise NotImplementedError()
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

    def apply_model(self, x_noisy, t, cond, is_training=False, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1) # b, l, c
        # cond_index = torch.cat(cond['c_index'], 1) # b, 1
        vc_feats = cond.get('vc_feats', None)
        if vc_feats is None:
            cond_index = cond['c_index'][0]  # b, t
            cond_vc = torch.cat(cond['c_video'], 2)  # b, c, t, h, w
            vc_feats, kl_loss = self.videocontent_enc(x=cond_vc, index=cond_index,
                                                      is_training=is_training)
        eps = diffusion_model(x=x_noisy, vc_feats=vc_feats, timesteps=t, context=cond_txt)

        if is_training:
            return eps, kl_loss
        else:
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
            samples, z_denoise_row = self.sample_log(cond=new_cond, verbose=verbose,
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
    def log_videos(self, batch, N=8, n_frames=4, sample=True, ddim_steps=50, ddim_eta=0.0,
                   verbose=False, unconditional_guidance_scale=3.0, **kwargs):
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
        log["reconstruction"] = self.decode_first_stage(z)[:N, :]
        log['full_frames'] = batch[self.videocontent_key][:N, :].to(self.device)
        # start sampling
        assert unconditional_guidance_scale >= 0.0
        if sample or unconditional_guidance_scale == 0.0:
            samples = []
            for f in td:
                ind = torch.ones_like(ind) * f / n_frames
                new_cond["c_index"] = [ind]
                frames, _ = self.sample_log(cond=new_cond, batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose)
                samples.append(self.decode_first_stage(frames).unsqueeze(1))
            x_samples = torch.cat(samples, dim=1) # b, t, c, h, w
            x_samples = rearrange(x_samples, 'b t c h w -> b c t h w')
            log["samples"] = x_samples

        if unconditional_guidance_scale > 0.0:
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
                curframe = self.decode_first_stage(frames_ug).unsqueeze(1)
                samples_ug.append(curframe)
            x_samples_ug = torch.cat(samples_ug, dim=1)  # b, t, c, h, w
            x_samples_ug = rearrange(x_samples_ug, 'b t c h w -> b c t h w')
            log[f"samples_ug_scale_{unconditional_guidance_scale:.2f}"] = x_samples_ug
        return log

    @torch.no_grad()
    def parallel_sample(self, batch, vidcontent, n_frames=4,
                        ddim_steps=50, ddim_eta=0.0, unconditional_guidance_scale=3.0, verbose=False, **kwargs):
        N = batch[self.frame_index_key].shape[0]
        use_ddim = ddim_steps is not None
        z, cond = self.get_input(batch, None, repeat_c_by_T=False)
        # obtain textual and index conditions
        c = cond["c_crossattn"][0] # b, 77, 1024
        c = repeat(c, 'b c l -> b n c l', n=n_frames)
        c = rearrange(c, 'b n c l -> (b n) c l')
        index = []
        for t in range(n_frames):
            cind = torch.ones((N, 1)) * t / n_frames
            index.append(cind.to(self.device))
        index = torch.cat(index, dim=0) # (b n) 1
        # obtain condition vc_feats
        vidcontent = repeat(vidcontent, 'b c h w -> b n c h w', n=n_frames)
        vidcontent = rearrange(vidcontent, 'b n c h w -> (b n) c h w')
        vc_feats = self.videocontent_enc.decode(vidcontent, index)
        new_cond = {"c_crossattn": [c], "vc_feats": vc_feats}
        # obtain uncondition conditions
        uc_cross = self.get_unconditional_conditioning(N*n_frames)
        input_frame = torch.zeros_like(batch['full_frame'])
        input_frame = rearrange(input_frame, 'b c t h w -> (b t) c h w')
        latent_x = self.encode_first_stage(input_frame.cuda())
        latent_x = self.get_first_stage_encoding(latent_x)
        latent_x = rearrange(latent_x, '(b t) c h w -> b c t h w', b=N)
        uc_vidcontent = self.videocontent_enc.encode(latent_x).mode()
        uc_vidcontent = repeat(uc_vidcontent, 'b c h w -> b n c h w', n=n_frames)
        uc_vidcontent = rearrange(uc_vidcontent, 'b n c h w -> (b n) c h w')
        uc_feats = self.videocontent_enc.decode(uc_vidcontent, index)
        uc_cond = {"c_crossattn": [uc_cross], "vc_feats": uc_feats}
        # start sampling
        assert unconditional_guidance_scale >= 0.0
        if unconditional_guidance_scale == 0.0:
            frames, _ = self.sample_log(cond=new_cond, batch_size=N*n_frames, ddim=use_ddim,
                                         ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose)
            x_samples = self.decode_first_stage(frames)
            x_samples = rearrange(x_samples, '(b n) c h w -> b c n h w', n=n_frames)
        else:
            frames_ug, _ = self.sample_log(cond=new_cond, batch_size=N*n_frames, ddim=use_ddim,
                                           ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose,
                                           unconditional_conditioning=uc_cond,
                                           unconditional_guidance_scale=unconditional_guidance_scale)
            x_samples = self.decode_first_stage(frames_ug)
            x_samples = rearrange(x_samples, '(b n) c h w -> b c n h w', n=n_frames)
        return x_samples



    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, verbose=False, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     cond, verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        if self.allow_gan:
            return self.configure_optimizers_wgan()
        else:
            return self.configure_optimizers_wogan()

    def configure_optimizers_wogan(self):
        lr = self.learning_rate
        params = list(self.videocontent_enc.parameters())

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
                params.append(p)

        # count total params to optimize
        optimize_params = 0
        for param in params:
            optimize_params += param.numel()
        print(f"NOTE!!! {optimize_params/1e6:.3f}M params to optimize in TOTAL!!!")

        if self.optimizer == "hybridadam":
            raise NotImplementedError
        else:
            print("Load AdamW optimizer.")
            opt = torch.optim.AdamW(params, lr=lr)

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

    def configure_optimizers_wgan(self):
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
                # print("Add optimize parameters: ", n)
                g_params.append(p)

        # == parameters of discriminator ==
        d_params = []
        d_params += list(self.discriminator.parameters())
        assert len(d_params) != 0

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

        if self.generator_frequency is None or self.discriminator_frequency is None:
            print(f"- NOTE: Train without FREQUENCY!")
            return [opt_g, opt_d]
        else:
            print(f"- NOTE: Train with FREQUENCY g{self.generator_frequency}/d{self.discriminator_frequency}!")
            return (
                {'optimizer': opt_g, 'frequency': self.generator_frequency},
                {'optimizer': opt_d, 'frequency': self.discriminator_frequency}
            )

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
