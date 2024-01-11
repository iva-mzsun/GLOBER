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
# Train scripts for both the auto-encoder and generator are the same
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base $CFG --logdir experiments/ # --ckpt path/to/ckpt
```

## Sample&Evaluation Scripts
We follow the implementation of StyleGAN-V(https://github.com/universome/stylegan-v) for evaluation.
```
# AutoEncoder
bash scripts/script_for_sample_3c.sh $CUR $CUDA $TOTAL $CFG $EXP $PTH $UC_FRAME $UC_VIDEO $UC_DOMAIN
bash scripts/script_for_fvd_3c.sh $EXP $UCFRAME $UCVID $UCDOMAIN $PTH $CUDA

# Generator
bash scripts/script_for_sample.sh $CFG $EXP $PTH $UC $CUR $TOTAL $CUDA
bash scripts/script_for_fvd.sh $EXP $UC $PTH $CUDA
```

## Checkpoints
Will be released soon.

## Test generation speed of prior methods

VIDM: \url{https://github.com/MKFMIKU/vidm}

VDM: \url{https://github.com/lucidrains/video-diffusion-pytorch}

VideoFusion: \url{https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video}

TATS:  \url{https://github.com/SongweiGe/TATS}
