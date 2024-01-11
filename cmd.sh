#!/bin/bash

# train video auto-encoder
CFG=configs/SkyTimelapse/sky_ldmae_v0.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logdir experiments/ --base $CFG  --debug True --ngpu 8
#--ckpt

