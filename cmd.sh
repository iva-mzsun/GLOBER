CFG=configs/UCF101/ldmaev3_attempt_optimize_params/ldm_ae_v3-4-5-2.yaml
#EXP=?
# 训练命令
#TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py --logdir experiments/ --base $CFG --ckpt models/control_sd21_ini.ckpt --debug True --ngpu 1
python main.py --logdir experiments/ --base $CFG --debug True --ngpu 1