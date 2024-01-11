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

from utils.util import instantiate_from_config, tensor2img
# from utils.modules.attention import enable_flash_attentions
from main import get_parser, load_state_dict, nondefault_trainer_args

def get_dataloader(data_cfg, batch_size, shuffle=False):
    print(f"Shuffle datasets : {shuffle}")
    import torch.utils.data as Data
    # data_cfg.params.shuffle = shuffle
    dataset = instantiate_from_config(data_cfg)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        drop_last=False
    )
    print(data_cfg)
    print(f"- len(dataset): {len(dataset)}")
    print(f"- len(dataloader): {len(dataloader)}")
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
        paths = opt.resume.split("/")
        logdir = "/".join(paths[:-2])
        rank_zero_info("logdir: {}".format(logdir))
        ckpt = opt.resume

        # base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        # opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        logdir = opt.logdir
        ckpt = "nan"

    # init and save configs
    configs = [OmegaConf.load(cfg.strip()) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())

    ddim_sampler = opt.ddim_sampler or config.model["params"]["ddim_sampler"]
    config.model["params"]["ddim_sampler"] = ddim_sampler
    sampledir = os.path.join(logdir, f'samples-{opt.save_mode}-{opt.ddim_sampler}-'
                                     f'f{opt.ucgs_frame}-v{opt.ucgs_video}-d{opt.ucgs_domain}-'
                                     f'{os.path.basename(ckpt).split(".")[0]}')
    if opt.suffix != "":
        sampledir = sampledir + '-' + opt.suffix
    os.makedirs(sampledir, exist_ok=True)
    seed_everything(opt.seed)

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
    model.register_schedule(linear_start=model.linear_start, linear_end=model.linear_end)
    model = model.cuda().eval()

    # data
    print(f"- Loading validation data...")
    bs = opt.batch_size or config.data.params.batch_size
    if opt.caps_path is not None:
        config.data.params.validation.target = "data.custom.VideoFolderDataset_Inference"
        config.data.params.validation.params.caps_path = opt.caps_path
        config.data.params.validation.params.num_replication = opt.num_replication
    if opt.dataset_root is not None:
        config.data.params.validation.params.root = opt.dataset_root
    config.data.params.validation.params.max_data_num = opt.total_sample_number
    dataloader = get_dataloader(config.data.params.validation, bs, opt.shuffle)
    part_num = len(dataloader) / opt.total_part
    start_idx = int((opt.cur_part - 1) * part_num)
    end_idx = int(opt.cur_part * part_num)

    # start to generate
    vc = None
    ddim_step = opt.ddim_step
    save_mode = opt.save_mode  # bybatch, byvideo, byframe
    verbose = opt.test_verbose
    video_length = opt.video_length
    total_sample_number = opt.total_sample_number
    use_ddim = opt.use_ddim and (ddim_step > 0)
    print(f"- Saving generated samples to {sampledir}")
    print(f"- Use ddim: {use_ddim} with ddim steps: {ddim_step}")
    print(f"- Cur part {opt.cur_part}/{opt.total_part} with idx from {start_idx} to {end_idx}")
    batch_idx = 0
    for batch in tqdm(iter(dataloader), desc=f"new ucgs: {opt.new_unconditional_guidance}"):
        if batch_idx >= start_idx and batch_idx < end_idx:
            if len(os.listdir(sampledir)) >= total_sample_number:
                final_number = max(len(os.listdir(sampledir)), bs * (batch_idx + 1))
                print(f"Having generated {final_number} video samples in {sampledir}!")
                exit()

            start_t = time.time()
            ucgs_frame = opt.ucgs_frame
            ucgs_video = opt.ucgs_video
            ucgs_domain = opt.ucgs_domain
            sample_log = model.log_videos_parallel(batch, N=bs, n_frames=video_length, verbose=verbose,
                                                   sample=False, use_ddim=use_ddim, ddim_steps=ddim_step,
                                                   ucgs_frame=ucgs_frame, ucgs_video=ucgs_video, ucgs_domain=ucgs_domain)
            end_t = time.time()
            # print(f"Generation time: {end_t - start_t}s")
            cur_video = sample_log[f"samples_ug_scale"].detach().cpu() # b c t h w
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
                        # frame.save(os.path.join(save_path, f"{t:04d}.png"))
                        frame.save(os.path.join(save_path, f"{t:04d}.jpg"))
            else:
                raise NotImplementedError

        batch_idx += 1
        torch.cuda.empty_cache()
