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
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except:
    import pytorch_lightning as pl

try:
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "lightning.pytorch."
except:
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    LIGHTNING_PACK_NAME = "pytorch_lightning."

from ipdb import set_trace as st

from ldm.util import instantiate_from_config, tensor2img
# from ldm.modules.attention import enable_flash_attentions

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'hint' in k:
            continue
        new_state_dict[k] = v
    print(f'Loaded state_dict from [{ckpt_path}]')
    return new_state_dict

def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="running mode",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default="",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--ckpt",
        default='./models/control_sd21_ini.ckpt'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=None
    )
    parser.add_argument(
        "--logger_freq",
        type=int,
        default=300
    )
    parser.add_argument(
        "--sd_locked",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--only_mid_control",
        type=bool,
        default=False
    )

    # for generation
    parser.add_argument("--save_mode", type=str, default="byvideo") # bybatch, byvideo, byframe
    parser.add_argument("--test_verbose", type=bool, default=False)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--total_sample_number", type=int, default=16)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=9.0)

    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    # dataset = worker_info.dataset
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)

class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(self.datasets["train"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=True,
                           worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(self.datasets["validation"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn,
                           shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        return DataLoaderX(self.datasets["test"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn,
                           shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(self.datasets["predict"],
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           worker_init_fn=init_fn)

class SetupCallback(Callback):

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # def on_pretrain_routine_start(self, trainer, pl_module):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Experiment directories:")
            print(f" - logdir: {self.logdir}")
            print(f" - cfgdir: {self.cfgdir}")
            print(f" - ckptdir: {self.ckptdir}")

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    # def on_fit_end(self, trainer, pl_module):
    #     if trainer.global_rank == 0:
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         rank_zero_info(f"Saving final checkpoint in {ckpt_path}.")
    #         trainer.save_checkpoint(ckpt_path)

class ImageLogger(Callback): # Report ERRORS under ColossalAI strategy.

    def __init__(self,
                 clamp=True,
                 rescale=True,
                 local_dir=None,
                 log_imgs=False,
                 disabled=False):
        super().__init__()
        self.rescale = rescale
        self.clamp = clamp
        self.disabled = disabled
        self.log_imgs = log_imgs
        self.local_dir = local_dir
        if self.local_dir:
            os.makedirs(local_dir, exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int):
        if not self.disabled and pl_module.global_step > 0 and batch_idx == 0:
            sample_logs = outputs
            for k in sample_logs.keys():
                img = tensor2img(sample_logs[k])
                # save logger
                if self.log_imgs:
                    trainer.logger.experiment.log({k: [wandb.Image(img)]})
                # save local
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, pl_module.global_step,
                                                                    pl_module.current_epoch, batch_idx)
                try:
                    path = os.path.join(self.local_dir, filename)
                    img.save(path)
                except:
                    pass

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py

    def on_train_start(self, trainer, pl_module):
        rank_zero_info("Training is starting")

    def on_train_end(self, trainer, pl_module):
        rank_zero_info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

def obtain_trainer_kwargs(trainer_config, opt):
    # trainer and callbacks
    trainer_kwargs = dict()

    # config the logger
    default_logger_cfgs = {
        "wandb": {
            "target": LIGHTNING_PACK_NAME + "loggers.WandbLogger",
            "params": {
                "project": "control",
                "name": nowname,
                "save_dir": "experiments/",
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "tensorboard": {
            "target": LIGHTNING_PACK_NAME + "loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "diff_tb",
                "log_graph": True
            }
        }
    }
    logger_cfg = default_logger_cfgs["wandb"]
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # config the strategy, defualt is ddp
    if "strategy" in trainer_config:
        strategy_cfg = trainer_config["strategy"]
        strategy_cfg["target"] = LIGHTNING_PACK_NAME + strategy_cfg["target"]
    else:
        strategy_cfg = {
            "target": LIGHTNING_PACK_NAME + "strategies.DDPStrategy",
            "params": {
                "find_unused_parameters": False
            }
        }
    trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "clamp": True,
                "local_dir": imgdir,
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "main.CUDACallback"
        },
    }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": LIGHTNING_PACK_NAME + "callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:04}-{step:06}",
            "verbose": True,
            # "save_last": True,
            "every_n_epochs": 1,
            "save_on_train_epoch_end": True,
        }
    }
    if hasattr(model, "monitor"):
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 2
    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    callbacks_cfg['model_ckpt'] = modelckpt_cfg

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    return trainer_kwargs

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
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
            ckpt = opt.ckpt.strip()

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    imgdir = os.path.join(logdir, "images")
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
    load_strict = trainer_config.get('ckpt_load_strict', True)
    model = instantiate_from_config(config.model).cpu()
    model.load_state_dict(load_state_dict(ckpt.strip(), location='cpu'), strict=load_strict)
    print(f"Load ckpt from {ckpt} with strict {load_strict}")

    # create trainer
    trainer_kwargs = obtain_trainer_kwargs(trainer_config, opt)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # data
    print(f"- Loading data...")
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

    for k in data.datasets:
        rank_zero_info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    print(f"- Configuring learning rate...")
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    model.learning_rate = base_lr
    rank_zero_info("++++ NOT USING LR SCALING ++++")
    rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)
    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.mode == 'train':
        try:
            trainer.fit(model, data)
        except Exception:
            if opt.debug is False:
                melk()
            raise
    elif opt.mode == 'generate':
        trainer.test(model, data)

