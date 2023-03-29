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
    data_cfg.params.shuffle = False
    dataset = instantiate_from_config(data_cfg)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
    # sampledir = os.path.join(logdir, f'samples-{opt.save_mode}-ucgs{opt.unconditional_guidance_scale}-'
    #                                  f'{os.path.basename(ckpt).split(".")[0]}')
    # os.makedirs(sampledir, exist_ok=True)
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
        config.data.params.train.params.root = opt.dataset_root
        config.data.params.validation.params.root = opt.dataset_root
    dataloader = get_dataloader(config.data.params.train, bs)

    vcs = []
    for e in tqdm(range(20)): # epoch
    # for e in tqdm(range(12)): # epoch
        for batch in tqdm(iter(dataloader)):
            vc, _ = model.get_input(batch, None)
            vcs.append(vc)

    allvc = torch.cat(vcs, dim=0)
    std = torch.std(allvc, dim=0).detach().cpu().numpy()
    mean = torch.mean(allvc, dim=0).detach().cpu().numpy()
    np.save('notebook/sample_gauss_shifts/ucf101_256x_std.npy', std)
    np.save('notebook/sample_gauss_shifts/ucf101_256x_mean.npy', mean)
