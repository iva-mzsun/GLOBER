import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config
from einops import rearrange, repeat
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 use_fp16,
                 stable_ae_config,
                 videocontent_key,
                 vc_size, vc_channels,
                 latent_size, latent_channels,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        for p in model.parameters():
            p.requires_grad = False
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z.half() if self.use_fp16 else self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_input(self, batch):
        # obtain & encode x
        # x = batch[self.first_stage_key]
        # assert len(x.shape) == 5 # b, c, t, h, w
        # x = rearrange(x, 'b c t h w -> (b t) c h w')
        # if self.use_fp16:
        #     x = x.to(memory_format=torch.contiguous_format).half()
        # else:
        #     x = x.to(memory_format=torch.contiguous_format).float()
        # x = x.to(self.device)
        # encoder_posterior = self.first_stage_model.encode(x)
        # z = self.get_first_stage_encoding(encoder_posterior).detach()

        # obtain & encode full video frames
        vidcontent = batch[self.videocontent_key].to(self.device)
        if self.use_fp16:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).half()
        else:
            vidcontent = vidcontent.to(memory_format=torch.contiguous_format).float()
        T = vidcontent.shape[2]
        vidcontent = rearrange(vidcontent, 'b c t h w -> (b t) c h w')
        vidcontent = self.first_stage_model.encode(vidcontent)
        vidcontent = self.get_first_stage_encoding(vidcontent).detach()
        vidcontent = rearrange(vidcontent, '(b t) c h w -> b c t h w', t=T)
        vc = self.stable_ae.encode(vidcontent).detach() # b c h w

        # obtain c
        cond_key = self.cond_stage_key
        if cond_key in ['caption', 'coordinates_bbox', "txt"]:
            xc = batch[cond_key]
        else:
            raise NotImplementedError
        if isinstance(xc, dict) or isinstance(xc, list):
            c = self.get_learned_conditioning(xc)
        else:
            c = self.get_learned_conditioning(xc.to(self.device))

        # out
        out = [vc, c]
        return out

# =============================
    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log





    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer