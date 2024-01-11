import einops
import torch
import torch as th
import torch.nn as nn
from einops import rearrange, repeat

from utils.util import instantiate_from_config, get_obj_from_str
from ldm_autoencoder.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm_autoencoder.modules.diffusionmodules import UNetModel, \
    TimestepEmbedSequential, ResBlock, Downsample, normalization, ResBlock2n
from ldm_autoencoder.modules.diffusionmodules.util import conv_nd, linear, \
    zero_module, timestep_embedding

from ldm_autoencoder.diffusion.ddim_v1 import DDIMSampler
from ldm_autoencoder.diffusion.ddpm_v1 import LatentDiffusion
from ldm_autoencoder.utils.blocks import ResBlockwoEmb, TemporalAttentionBlock, \
    SpatialAttentionBlock, ResBlockwIndwEmb, IndPrexEmbedSequential


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
        learnvar=True
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.learnvar = learnvar
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
            if self.learnvar:
                self.out_vc_var = nn.Sequential(
                    conv_nd(dims, ch, self.out_channels, 3, padding=1),
                    normalization(self.out_channels),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, self.out_channels, self.out_channels, 1))
                )
        else:
            self.out_vc = nn.Sequential(
                conv_nd(dims, ch, self.out_channels, 3, padding=1),
                normalization(self.out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, image_size // ds, image_size // ds)),
                conv_nd(dims, self.out_channels, self.out_channels, 1)
            )
            if self.learnvar:
                self.out_vc_var = nn.Sequential(
                    conv_nd(dims, ch, self.out_channels, 3, padding=1),
                    normalization(self.out_channels),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool3d((1, image_size // ds, image_size // ds)),
                    zero_module(conv_nd(dims, self.out_channels, self.out_channels, 1))
                )

        # embed index embed
        index_embed_dim = model_channels * 4
        self.index_embed = nn.Sequential(
            linear(model_channels, index_embed_dim),
            nn.SiLU(),
            linear(index_embed_dim, index_embed_dim),
        )

        # obtain output blocks
        self.sdinput_block_ds = sdinput_block_ds
        self.sdinput_block_chans = sdinput_block_chans
        self.output_blocks = nn.ModuleList([])
        self.exchange_blocks = nn.ModuleList([])
        for i, ch in enumerate(sdinput_block_chans):
            curds, n = ds, 0
            if i == 0:
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
            else:
                tards = sdinput_block_ds[i]
                preds = sdinput_block_ds[i-1]
                if tards == preds:
                    continue # skip this layer
                emb_down = False if preds == tards else True
                if tards == ds:
                    layers = [ResBlockwIndwEmb(out_channels, index_embed_dim, prech,
                                               dropout, ch, emb_down=emb_down)]
                elif ds < tards:
                    while curds < tards:
                        n += 1
                        curds *= 2
                    layers = [ResBlockwIndwEmb(out_channels, index_embed_dim, prech,
                                               dropout, ch, down=True, sample_num=n, emb_down=emb_down)]
                else:
                    while curds > tards:
                        n += 1
                        curds /= 2
                    layers = [ResBlockwIndwEmb(out_channels, index_embed_dim, prech,
                                               dropout, ch, up=True, sample_num=n, emb_down=emb_down)]
                layers.append(self.make_zero_conv(ch, dims=2))
                self.output_blocks.append(IndPrexEmbedSequential(*layers))
            # stream of temporal information
            if num_head_channels == -1:
                dim_head = ch // num_heads
            else:
                num_heads = ch // num_head_channels
                dim_head = num_head_channels
            if legacy:
                dim_head = num_head_channels
            frame_exchange_layer = TemporalAttentionBlock(
                                        ch,
                                        use_checkpoint=use_checkpoint,
                                        num_heads=num_heads,
                                        num_head_channels=dim_head,
                                        use_new_attention_order=use_new_attention_order,
                                    )
            self.exchange_blocks.append(frame_exchange_layer)
            prech = ch

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
            h_mean = self.out_vc(h)
            vc_mean = h_mean[:, :, 0, :, :]  # B, C, H, W
            if self.learnvar:
                h_std = self.out_vc_var(h)
                vc_std = h_std[:, :, 0, :, :]
        else:
            vc_mean = self.out_vc(h)  # B, C, 1, H, W
            vc_mean = vc_mean.squeeze(2)  # B, C, H, W
            if self.learnvar:
                vc_std = self.out_vc_var(h)
                vc_std = vc_std.squeeze(2)
        if self.learnvar:
            vc_dist = torch.cat([vc_mean, vc_std], dim=1)
            vc_posterior = DiagonalGaussianDistribution(vc_dist)
            if self.training:
                global_feat = vc_posterior.sample()
                kl_loss = vc_posterior.kl()
            else:
                global_feat = vc_posterior.mode()
                kl_loss = None
        else:
            global_feat = vc_mean
            kl_loss = None
        return global_feat, kl_loss

    def decode(self, vc, full_index, tar_index):
        """
            vc: B, C, H, W
            index: B, T
        """
        # resize vc
        B, FullT = full_index.shape
        vc = repeat(vc, 'b c h w -> b c t h w', t=FullT)
        vc = rearrange(vc, 'b c t h w -> (b t) c h w')
        # obtain full index embedding
        index = rearrange(full_index, 'b t -> (b t)').to(vc.device)
        index_emb = timestep_embedding(index, self.model_channels)
        index_emb = self.index_embed(index_emb)
        # obtain tar index one-hot
        tar_index = (tar_index * (FullT - 1)).int().long()

        # obtain insert feat
        module_pos = 0
        vc_feats = dict({})
        pre_vc_feat = None
        for i, ds in enumerate(self.sdinput_block_ds):
            if i == 0:
                vc_feat = self.output_blocks[module_pos](vc, index_emb)
            elif ds == self.sdinput_block_ds[i - 1]:
                continue # skip this layer
            else:
                vc_feat = self.output_blocks[module_pos](vc, index_emb, pre_vc_feat)
            pre_vc_feat = vc_feat
            _, C, H, W = vc_feat.shape
            vc_feat = rearrange(vc_feat, '(b ft) c h w -> b c ft h w', ft=FullT)
            vc_feat = self.exchange_blocks[module_pos](vc_feat)
            cur_index = repeat(tar_index, 'b t -> b c t h w', c=C, h=H, w=W)
            vc_feat = vc_feat.gather(2, cur_index)
            vc_feat = rearrange(vc_feat, 'b c t h w -> (b t) c h w')
            vc_feats[i] = vc_feat
            module_pos += 1
        return vc_feats

class AEUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super(AEUnetModel, self).__init__(*args, **kwargs)
        # mapping noisy first frame
        # self.process_frame = zero_module(
        #     conv_nd(self.dims, self.in_channels * 2, self.model_channels, 3, padding=1))

    def forward(self, x, preframe_noisy=None,
                timesteps=None, context=None, vc_feats=None,
                **kwargs):
        # obtain timestep embeddings
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            t_emb = self.time_embed(t_emb)
        # sum embeddings
        emb = t_emb

        # process first frame
        # print(x.shape, preframe_noisy.shape)
        # frame = torch.cat([x, preframe_noisy], dim=1)
        # preframe = self.process_frame(frame)

        # encode: insert vc feats
        hs = []
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            # if i == 0:
            #     h = h + preframe
            if vc_feats.get(i, None) is not None:
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
                 preframe_index_key, preframe_key,
                 frame_index_key, full_index_key, optimize_params_key,
                 allow_gan, gan_config,
                 kl_loss_weight=1e-6, optimizer="adam",
                 ddim_sampler="ddim_v1",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.kl_loss_weight = kl_loss_weight
        self.full_index_key = full_index_key
        self.frame_index_key = frame_index_key
        self.optimize_params_key = optimize_params_key
        self.ddim_target = f"ldm_autoencoder.diffusion.{ddim_sampler}.DDIMSampler"
        # video encoder: encode video key frames to global feature
        self.videocontent_key = videocontent_key
        videoenc_config['params']['sdinput_block_ds'] = self.model.diffusion_model.input_block_ds
        videoenc_config['params']['sdinput_block_chans'] = self.model.diffusion_model.input_block_chans
        self.videocontent_enc = instantiate_from_config(videoenc_config)
        # obtain target video frame indexes
        self.frame_index_key = frame_index_key
        self.preframe_key = preframe_key
        self.preframe_index_key = preframe_index_key
        # Dataset Condition: Initialize learnable embedding for current dataset
        self.context_emb = nn.Parameter(torch.randn(64, 768) * 0.1)
        # Video Condition: Projecting global feature
        self.global_projector = nn.Sequential(
            nn.Conv2d(self.videocontent_enc.out_channels, 768,
                      kernel_size=7, padding=3, stride=2),
            nn.SiLU(),
            nn.Conv2d(768, 768, kernel_size=3, padding=1)
        )
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
        self.use_input_frames = cfg.get('use_input_frames', False)
        self.decode_z_to_frames = cfg.get('decode_z_to_frames', False)
        self.generator_frequency = cfg.generator_frequency
        self.discriminator_frequency = cfg.discriminator_frequency
        self.generator_loss_weight = cfg.generator_loss_weight
        self.discriminator = instantiate_from_config(cfg.discriminator)

    def obtain_disc_label(self, batch, realx, fakex, optimizer_idx):
        ''' Ensure indexes of inputs are sorted!!!
        realx/fakex: [B * T, C, H, W]
        '''
        B, T = batch[self.frame_index_key].shape
        if self.decode_z_to_frames:
            if self.use_input_frames:
                realx = batch[self.first_stage_key][:B]
                realx = rearrange(realx, 'b c t h w -> (b t) c h w')
                realx = realx.to(memory_format=torch.contiguous_format)
                realx = realx.half() if self.use_fp16 else realx.float()
            else:
                realx = self.decode_first_stage(realx)
            fakex = self.decode_first_stage(fakex)
        realx = rearrange(realx, '(b t) c h w -> b c t h w', b=B, t=T)
        fakex = rearrange(fakex, '(b t) c h w -> b c t h w', b=B, t=T)
        real0 = realx[:, :, :1, :, :]  # B, C, 1, H, W
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
        elif optimizer_idx == 1:  # train discriminator
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
        samples = torch.cat(samples, dim=0)  # (b * 3 or 4) c 2 h w
        labels = torch.cat(labels, dim=0).to(samples.device)  # (b * 3 or 4)
        return samples, labels

    def forward(self, x, c, is_training=False, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, is_training=is_training, *args, **kwargs)

    def shared_step(self, batch, is_training=False, **kwargs):
        x, cond = self.get_input(batch, self.first_stage_key)
        loss = self(x, cond, is_training=is_training, **kwargs)
        return loss

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
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            # obtain temporal discriminator results
            for p in self.discriminator.parameters():
                p.requires_grad = True
            samplex, labels = self.obtain_disc_label(batch, x_start, x_recon, 0)
            logits = self.discriminator(samplex)
            g_loss = self.adversarial_loss(logits, labels)
            self.log("train/g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
            self.log("train/d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return d_loss

    def p_losses(self, x_start, cond, t, noise=None, is_training=False, returnx=False):
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # obtain noisy pre frames and corresponding common noises
        # print(cond['pre_frames'].shape, cond['pre_frame_mask'].shape)
        pre_mask = cond['pre_frame_mask']
        frame_pre = cond['pre_frames']
        noise_pre = torch.randn_like(frame_pre)
        noisy_frame_pre = self.q_sample(x_start=frame_pre,
                                        t=t, noise=noise_pre)
        noisy_frame_pre = rearrange(noisy_frame_pre,
                                    '(b t) c h w -> b t c h w',
                                    b=pre_mask.shape[0])
        uc_frame_pre = torch.zeros_like(noisy_frame_pre)
        noisy_frame_pre = uc_frame_pre * pre_mask[:, None, None, None, None] + \
                          noisy_frame_pre * (1 - pre_mask[:, None, None, None, None])
        noisy_frame_pre = rearrange(noisy_frame_pre, 'b t c h w -> (b t) c h w')
        cond['noisy_frame_pre'] = noisy_frame_pre
        # obtain noisy target frame
        noise_tar = torch.randn_like(x_start) # [bt c h w]
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_tar)

        if is_training is True:
            model_output, kl_loss = self.apply_model(x_noisy, t, cond, is_training)
        else:
            kl_loss = None
            model_output = self.apply_model(x_noisy, t, cond, is_training)

        if self.parameterization == "eps":
            target = noise_tar
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
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
            else:
                raise NotImplementedError()
            return loss, loss_dict, x_start, x_recon
        return loss, loss_dict

    @torch.no_grad()
    def get_video_frames(self, batch, k, keep_tdim=True, bs=None):
        # obtain original video frames
        frames = batch[k].to(self.device) # [b c t h w]
        if bs is not None:
            frames = frames[:bs]
        if self.use_fp16:
            frames = frames.to(memory_format=torch.contiguous_format).half()
        else:
            frames = frames.to(memory_format=torch.contiguous_format).float()
        T = frames.shape[2]
        frames = einops.rearrange(frames, 'b c t h w -> (b t) c h w')
        frames = self.encode_first_stage(frames)
        frames = self.get_first_stage_encoding(frames).detach()
        if keep_tdim:
            frames = rearrange(frames, '(b t) c h w -> b c t h w', t=T)
        return frames

    def get_input(self, batch, k, bs=None, repeat_c_by_T=True, *args, **kwargs):
        with torch.no_grad():
            # obtain target video frames
            x = self.get_video_frames(batch, k=self.first_stage_key,
                                      keep_tdim=False, bs=bs)
            # encode video frames
            key_frames = self.get_video_frames(batch, k=self.videocontent_key,
                                                keep_tdim=True, bs=bs) # [b c n h w]
            pre_frames = self.get_video_frames(batch, k=self.preframe_key,
                                               keep_tdim=False, bs=bs)  # [bt c h w]
            multi = x.shape[0] // key_frames.shape[0]

            full_indexes = batch[self.full_index_key].to(self.device)
            tar_indexes = batch[self.frame_index_key].to(self.device)
            if bs is not None:
                tar_indexes = tar_indexes[:bs]
                full_indexes = full_indexes[:bs]

        # obtain context embeddings
        c = repeat(self.context_emb, 'l c -> b l c', b=key_frames.shape[0])

        if self.training:
            # -------- domain-level conditions --------
            mask0 = (torch.rand(c.shape[0]).to(self.device) < 0.1) * 1.
            uc = torch.zeros_like(c)
            c = uc * mask0[:, None, None] + c * (1 - mask0[:, None, None])

            with torch.no_grad():
                # ---------- video-level conditions ---------
                mask2 = (torch.rand(c.shape[0]).to(self.device) < 0.1) * 1.
                uc_key_frames = torch.zeros_like(key_frames)
                key_frames = uc_key_frames * mask2[:, None, None, None, None] + \
                             key_frames * (1 - mask2[:, None, None, None, None])
                # ---------- frame-level conditions ---------
                mask1 = (torch.rand(c.shape[0]).to(self.device) < 0.1) * 1.
                # updata preframe condition
                pre_frames = rearrange(pre_frames, '(b t) c h w->b t c h w', t=multi)
                uc_preframe = torch.zeros_like(pre_frames)
                pre_frames = uc_preframe * mask1[:, None, None, None, None] + \
                             pre_frames * (1 - mask1[:, None, None, None, None])
                pre_frames = rearrange(pre_frames, 'b t c h w -> (b t) c h w')
        else:
            mask1 = (torch.rand(c.shape[0]).to(self.device) < 0.0) * 1.

        if repeat_c_by_T:
            multi = x.shape[0] // key_frames.shape[0]
            c = repeat(c, 'b l c -> b t l c', t=multi)
            c = rearrange(c, 'b t l c -> (b t) l c')
        # x: [bt c h w], c: [bt c h w], full_frames: [b c n h w]
        # frame_indexes: [b t], lamb: [bt]
        # pre_frame: [bt c h w], preframe_indexes: [b t]
        # first_frames: [bt c h w]
        return x, dict(prompt=c, key_frames=key_frames,
                       tar_index=tar_indexes, full_index=full_indexes,
                       pre_frames=pre_frames, pre_frame_mask=mask1)

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
        elif cond.get('global_feat', None) is None: # during training or testing
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

        context = context_emb + video_emb
        eps = diffusion_model(x=x_noisy, context=context,
                              vc_feats=frame_feats, timesteps=t,
                              preframe_noisy=cond_frame_pre)

        if self.training:
            return eps, kl_loss
        else:
            return eps

    @torch.no_grad()
    def log_videos(self, *args, **kwargs):
        return self.log_videos_parallel(sample=True, *args, **kwargs)

    @torch.no_grad()
    def log_videos_parallel(self, batch, N=8, n_frames=4, video_length=16, ddim_steps=50, ddim_eta=0.0, sample=True,
                            verbose=False, ucgs_domain=3.0, ucgs_frame=0.0, ucgs_video=4.0, **kwargs):
        # initialize settings
        log = dict()
        use_ddim = ddim_steps is not None
        N = min(batch[self.frame_index_key].shape[0], N)
        log['full_frames'] = batch[self.videocontent_key][:N]
        # obtain context embeddings
        c = repeat(self.context_emb, 'l c -> bt l c', bt=N*n_frames)
        uc = torch.zeros_like(c)
        # indexes of target video frames
        tar_index = [torch.ones((N, 1)) * t for t in range(n_frames)]
        tar_index = torch.cat(tar_index, dim=1).to(self.device) / n_frames  # b t
        full_index = [torch.ones((N, 1)) * t for t in range(video_length)]
        full_index = torch.cat(full_index, dim=1).to(self.device) / video_length  # b t

        # obtain conditional/unconditional inputs: vc feats
        key_frames = self.get_video_frames(batch, k=self.videocontent_key,
                                            keep_tdim=True, bs=N)
        global_feat, _ = self.videocontent_enc.encode(key_frames)
        frame_feats = self.videocontent_enc.decode(global_feat, full_index, tar_index)

        uc_key_frames = torch.zeros_like(key_frames)
        uc_global_feat, _ = self.videocontent_enc.encode(uc_key_frames)
        uc_frame_feats = self.videocontent_enc.decode(uc_global_feat, full_index, tar_index)

        # obtain noise with partly shared
        shape = (N, 1, self.channels, self.image_size, self.image_size)
        noise = [torch.randn(shape, device=self.device)]
        for i in range(n_frames - 1):
            cnoise = torch.randn(shape, device=self.device)
            noise.append(cnoise)
        noise = rearrange(torch.cat(noise, dim=1),
                          'b t c h w -> (b t) c h w')
        # collect conditional/unconditional inputs
        c_cond = dict(prompt=c, frame_feats=frame_feats, global_feat=global_feat)
        uc_cond_video = dict(prompt=c, frame_feats=uc_frame_feats, global_feat=uc_global_feat)
        uc_cond_domain = dict(prompt=uc, frame_feats=uc_frame_feats, global_feat=uc_global_feat)

        # start sampling
        if sample:
            torch.cuda.empty_cache()
            frames, _ = self.sample_log(cond=c_cond, ddim=use_ddim, x_T=noise,
                                        n_frame_per_video=n_frames, batch_size=N * n_frames,
                                        ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose)
            torch.cuda.empty_cache()
            x_samples = self.decode_first_stage(frames)
            x_samples = rearrange(x_samples, '(b n) c h w -> b c n h w', n=n_frames)
            log[f"samples"] = x_samples
        if ucgs_domain > 0 or ucgs_video > 0 or ucgs_frame > 0:
            torch.cuda.empty_cache()
            unconditional_guidance_scale = dict(domain=ucgs_domain, video=ucgs_video)
            unconditional_conditioning = dict(domain=uc_cond_domain, video=uc_cond_video)
            frames_ug, _ = self.sample_log(cond=c_cond, ddim=use_ddim, x_T=noise,
                                           n_frame_per_video=n_frames, batch_size=N * n_frames,
                                           ddim_steps=ddim_steps, eta=ddim_eta, verbose=verbose,
                                           unconditional_conditioning=unconditional_conditioning,
                                           unconditional_guidance_scale=unconditional_guidance_scale)
            torch.cuda.empty_cache()
            x_samples = self.decode_first_stage(frames_ug)
            x_samples = rearrange(x_samples, '(b n) c h w -> b c n h w', n=n_frames)
            log[f"samples_ug_scale"] = x_samples
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,
                   n_frame_per_video, x_T=None, verbose=False, **kwargs):
        # ddim_sampler = DDIMSampler(self)
        ddim_sampler = get_obj_from_str(self.ddim_target)(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape,
                                                     n_frame_per_video, cond, x_T=x_T, verbose=verbose, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        if self.allow_gan:
            return self.configure_optimizers_wgan()
        else:
            return self.configure_optimizers_wogan()

    def configure_optimizers_wogan(self):
        lr = self.learning_rate
        params = list(self.videocontent_enc.parameters())
        params += list(self.model.diffusion_model.parameters())
        params += list(self.global_projector.parameters())
        params += [self.context_emb]

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
        g_params += list(self.model.diffusion_model.parameters())
        g_params += list(self.global_projector.parameters())
        g_params += [self.context_emb]

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
