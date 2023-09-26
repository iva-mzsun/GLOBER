# BLOBER

The official code for the paper: 

GLOBER: Coherent Non-autoregressive Video Generation via Global Guided Video Decoder

[ARXIV](https://arxiv.org/abs/2309.13274) [DEMO](https://iva-mzsun.github.io/GLOBER)



## Conda Environment

```
conda env create -f environment.yaml
pip install -r requirements.txt
```

## Train Script
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base $CFG --logdir experiments/
```

## Sample Script
```
CUDA_VISIBLE_DEVICES=$CUDA python generate.py --base $CFG \
--resume experiments/$EXP/checkpoints/$PTH.ckpt \
--save_mode byframe --video_length 16 --total_sample_number 2048 \
--unconditional_guidance_scale $UC --ngpu 1 --batch_size 8 \
--parallel True --cur_part $CUR --total_part $TOTAL --seed $SEED
```

## Scipt to Test FVD score
We follow the implementation of StyleGAN-V(https://github.com/universome/stylegan-v).
```
CUDA_VISIBLE_DEVICES=$CUDA python src/scripts/calc_metrics_for_dataset.py \
--real_data_path $ROOT/$REAL_PATH \
--fake_data_path $ROOT/$FAKE_PATH \
--mirror 1 --gpus 1 --resolution $RESOLUTION \
--metrics fvd2048_16f --verbose 1 --use_cache 0
```

## Test generation speed of prior methods

VIDM: \url{https://github.com/MKFMIKU/vidm}

VDM: \url{https://github.com/lucidrains/video-diffusion-pytorch}

VideoFusion: \url{https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video}

TATS:  \url{https://github.com/SongweiGe/TATS}
