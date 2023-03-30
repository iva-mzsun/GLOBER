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
    # data_cfg.params.shuffle = False
    dataset = instantiate_from_config(data_cfg)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )
    return dataloader


if __name__ == "__main__":
    sys.path.append(os.getcwd())

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
        raise  NotImplementedError
    if opt.use_gauss_shift:
        sampledir = os.path.join(logdir, f'samples-{opt.save_mode}-ucgs{opt.unconditional_guidance_scale}-'
                                         f'{os.path.basename(ckpt).split(".")[0]}-shift')
    else:
        sampledir = os.path.join(logdir, f'samples-{opt.save_mode}-ucgs{opt.unconditional_guidance_scale}-'
                                         f'{os.path.basename(ckpt).split(".")[0]}')
    sampledir = sampledir + opt.suffix
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
    config.model["params"].update({"use_fp16": False})
    load_strict = trainer_config.get('ckpt_load_strict', True)
    model = instantiate_from_config(config.model).cpu()
    model.load_state_dict(load_state_dict(ckpt.strip(), location='cpu'), strict=load_strict)
    print(f"Load ckpt from {ckpt} with strict {load_strict}")
    model = model.cuda()

    # data
    print(f"- Loading validation data...")
    bs = opt.batch_size or config.data.params.batch_size
    if opt.dataset_root is not None:
        config.data.params.validation.params.root = opt.dataset_root
    dataloader = get_dataloader(config.data.params.validation, bs)
    part_num = len(dataloader) / opt.total_part
    start_idx = (opt.cur_part - 1) * part_num
    end_idx = opt.cur_part * part_num

    # start to generate
    vc = None
    if opt.use_gauss_shift:
        std = np.load(opt.gauss_shift_std)
        mean = np.load(opt.gauss_shift_mean)
        std = torch.tensor(std).to(model.device)
        mean = torch.tensor(mean).to(model.device)
    verbose = opt.test_verbose
    save_mode = opt.save_mode # bybatch, byvideo, byframe
    ddim_step = opt.ddim_step
    video_length = opt.video_length
    total_sample_number = opt.total_sample_number
    unconditional_guidance_scale = opt.unconditional_guidance_scale
    print(f"- Saving generated samples to {sampledir}")
    print(f"- Cur part {opt.cur_part}/{opt.total_part} with idx from {start_idx} to {end_idx}")
    batch_idx = 0
    for batch in tqdm(iter(dataloader)):
        if batch_idx >= start_idx and batch_idx < end_idx:
            if vc is None:
                vc, _ = model.get_input(batch, None)
            x_T = torch.randn_like(vc)
            if opt.use_gauss_shift:
                x_T = mean + x_T * std
            sample_log = model.log_videos(batch, N=bs, n_frames=video_length, x_T=x_T,
                                          verbose=verbose, sample=False, ddim_steps=ddim_step,
                                          unconditional_guidance_scale=unconditional_guidance_scale)
            if unconditional_guidance_scale == 0.0:
                cur_video = sample_log["samples"]
            else:
                cur_video = sample_log[f"samples_ug_scale_{unconditional_guidance_scale:.2f}"]
            cur_video = cur_video.detach().cpu() # b c t h w
            if save_mode == "bybatch":
                save = tensor2img(cur_video)
                save.save(os.path.join(sampledir, f"{batch_idx:04d}.jpg"))
            elif save_mode == "byvideo":
                video_names = batch['video_name']
                for b, name in enumerate(video_names):
                    save = tensor2img(cur_video[b].unsqueeze(0))
                    video_name = f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}"
                    save.save(os.path.join(sampledir, f"{video_name}.jpg"))
            elif save_mode == "byframe":
                video_names = batch['video_name']
                for b, name in enumerate(video_names):
                    video_name = f"b{batch_idx:04d}{b:02d}-v{name}-s{opt.seed}"
                    save_path = os.path.join(sampledir, video_name)
                    os.makedirs(save_path, exist_ok=True)
                    for t in range(video_length):
                        frame = tensor2img(cur_video[b, :, t, :, :])
                        frame.save(os.path.join(save_path, f"{t:04d}.jpg"))
            else:
                raise NotImplementedError

            if bs * (batch_idx + 1) >= total_sample_number or \
                    len(os.listdir(sampledir)) >= total_sample_number:
                final_number = max(len(os.listdir(sampledir)), bs * (batch_idx + 1))
                print(f"Having generated {final_number} video samples in {sampledir}!")
                exit()
        batch_idx += 1
