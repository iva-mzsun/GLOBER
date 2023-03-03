import argparse
import datetime
import glob
import os
import sys
import time
import wandb
import numpy as np
from functools import partial
from omegaconf import OmegaConf
from packaging import version
from prefetch_generator import BackgroundGenerator

import torch
from tqdm import tqdm
from einops import rearrange

try:
    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "lightning.pytorch."
except:
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "pytorch_lightning."

from ipdb import set_trace as st

from ldm.util import instantiate_from_config, tensor2img
# from ldm.modules.attention import enable_flash_attentions
from main import get_parser, load_state_dict, nondefault_trainer_args

def get_dataloader(data_cfg, batch_size):
    import torch.utils.data as Data
    dataset = instantiate_from_config(data_cfg)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True
    )
    return dataloader


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # set save directories
    if opt.resume:
        rank_zero_info("Resuming from {}".format(opt.resume))
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            rank_zero_info("logdir: {}".format(logdir))
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            rank_zero_info("Using base config {}".format(opt.base))
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name
        logdir = os.path.join(opt.logdir, nowname)
        if opt.ckpt:
            ckpt = opt.ckpt
    sampledir = os.path.join(logdir, f'samples-{os.path.basename(ckpt).split(".")[0]}')
    os.makedirs(sampledir, exist_ok=True)
    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg.strip()) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["devices"] = opt.ngpu or trainer_config["devices"]
    print(f"!!! WARNING: Number of gpu is {trainer_config['devices']} ")
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    use_fp16 = trainer_config.get("precision", 32) == 16
    if use_fp16:
        config.model["params"].update({"use_fp16": True})
    else:
        config.model["params"].update({"use_fp16": False})
    model = instantiate_from_config(config.model).cpu()
    model.load_state_dict(load_state_dict(ckpt.strip(), location='cpu'))
    model = model.cuda()

    # data
    print(f"- Loading validation data...")
    bs = opt.batch_size or config.data.params.batch_size
    dataloader = get_dataloader(config.data.params.validation, bs)

    # start to generate
    control_key = model.preframe_key
    verbose = opt.test_verbose
    save_mode = opt.save_mode # bybatch, byvideo, byframe
    video_length = opt.video_length
    total_sample_number = opt.total_sample_number
    unconditional_guidance_scale = opt.unconditional_guidance_scale

    for batch_idx, batch in enumerate(dataloader):
        # batch['tar_frame'] = None
        video_frames = [batch[control_key].detach().cpu()]
        for _ in tqdm(range(video_length), desc=f"generating {batch_idx}-th batch..."):
            # print(f"!!!!!!!!!! {_}")
            with torch.no_grad():
                sample_log = model.log_images(batch, N=bs, verbose=verbose,
                                              unconditional_guidance_scale=unconditional_guidance_scale)
            cur_frame = sample_log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]
            cur_frame = rearrange(torch.clamp(cur_frame, min=-1, max=1), 'b c h w -> b h w c')
            batch[control_key] = cur_frame
            video_frames.append(cur_frame.detach().cpu())

        cur_video = torch.cat(video_frames, dim=2) # concate on the width [B, C, H, W*T]
        if save_mode == "bybatch":
            save = rearrange(cur_video, 'b h w c -> c (b h) w')
            save = tensor2img(save)
            save.save(os.path.join(sampledir, f"{batch_idx:04d}.jpg"))
        elif save_mode == "byvideo":
            video_names = batch['video_name']
            for b, name in enumerate(video_names):
                save = rearrange(cur_video[b], 'h w c -> c h w')
                save = tensor2img(save)
                save.save(os.path.join(sampledir, # batchidx - videoname - seed
                                       f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}.jpg"))
        else:
            raise NotImplementedError

        if bs * (batch_idx + 1) >= total_sample_number:
            final_number = max(total_sample_number, batch * (batch_idx + 1))
            print(f"Having generated {final_number} video samples in {sampledir}!")
            exit()

