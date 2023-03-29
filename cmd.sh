#!/bin/bash
CFG=configs/UCF101/generator/test.yaml
# Debug
TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py --logdir experiments/ --base $CFG  --debug True --ngpu 1
#--ckpt

