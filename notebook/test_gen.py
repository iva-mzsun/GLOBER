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

def save_log_samples(batch_idx, batch, tensor, video_length, save_mode, sampledir):
    print(f"Current sampledir: {sampledir}")
    os.makedirs(sampledir, exist_ok=True)
    cur_video = tensor.detach().cpu()
    if save_mode == "bybatch":
        save = tensor2img(cur_video)
        save.save(os.path.join(sampledir, f"{batch_idx:04d}.jpg"))
    elif save_mode == "byvideo":
        video_names = batch['video_name']
        for b, name in enumerate(video_names):
            save = tensor2img(cur_video[b])
            video_name = f"b{batch_idx:04d}{b:02d}-v{name}"
            save.save(os.path.join(sampledir, f"{video_name}.jpg"))
    elif save_mode == "byframe":
        video_names = batch['video_name']
        for b, name in enumerate(video_names):
            video_name = f"b{batch_idx:04d}{b:02d}-v{name}"
            save_path = os.path.join(sampledir, video_name)
            os.makedirs(save_path, exist_ok=False)  # TODO
            sample = rearrange(cur_video[b], 'c h (t w) -> t c h w', t=video_length)
            for t in range(video_length):
                frame = tensor2img(sample[t])
                frame.save(os.path.join(save_path, f"{t:04d}.jpg"))
    else:
        raise NotImplementedError

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
    bs = 8
    # bs = opt.batch_size or config.data.params.batch_size
    if opt.dataset_root is not None:
        config.data.params.train.params.root = opt.dataset_root
        config.data.params.validation.params.root = opt.dataset_root
    dataloader = get_dataloader(config.data.params.validation, bs)

    """
     ====== test generation results ======
    """
    # hyper parameters
    ucgs = 9.0
    n_frames = 4
    save_mode = "bybatch"

    # obtain a batch as input
    print(f"- Logdir: {logdir}")
    batch_idx, batch = next(enumerate(dataloader))
    vc, cond = model.get_input(batch, None)
    noise = torch.randn_like(vc)

    # utils
    kwargs = {
        'batch': batch,
        'batch_idx': batch_idx,
        'save_mode': save_mode,
        'video_length': n_frames,
    }
    def sample_then_log(x_T, prefix):
        sample_log = model.log_videos(batch=batch, N=vc.shape[0], x_T=x_T,
                                      n_frames=n_frames, unconditional_guidance_scale=ucgs)
        tensor = sample_log["samples"]
        sampledir = os.path.join(logdir, f'{prefix}-s{opt.seed}')
        save_log_samples(tensor=tensor, sampledir=sampledir, **kwargs)
        tensor = sample_log[f"samples_ug_scale_{ucgs:.2f}"]
        sampledir = os.path.join(logdir, f'{prefix}-ucgs{ucgs:.2f}-s{opt.seed}')
        save_log_samples(tensor=tensor, sampledir=sampledir, **kwargs)

    print(batch['txt'])
    # ----- raw samples -----
    sample_then_log(x_T=noise, prefix="samples-raw")

    # ----- from training updates -----
    # state = torch.load('ucf_gauss_shifts.pth',
    #                    map_location=vc.device)['state_dict']
    # std = state['sample_gauss_std'].unsqueeze(0)
    # mean = state['sample_gauss_mean'].unsqueeze(0)
    std = np.load("notebook/sample_gauss_shifts/ucf101_256x_std_training.npy")
    mean = np.load("notebook/sample_gauss_shifts/ucf101_256x_mean_training.npy")
    std = torch.tensor(std).to(vc.device)
    mean = torch.tensor(mean).to(vc.device)
    sample_then_log(mean + std * noise, "samples-training")

    # ----- from independent process -----

    sample_then_log(mean + std * noise, "samples-preprocessing")