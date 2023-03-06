CFG=configs/UCF101/ldm_ae_v4.yaml
#EXP=?
# 训练命令
#TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py --logdir experiments/ --base $CFG --ckpt models/control_sd21_ini.ckpt --debug True --ngpu 1
python main.py --logdir experiments/ --base $CFG --debug True --ngpu 1 --ckpt experiments/2023-03-02T22-55-25_ldm_ae_v3-4/checkpoints/epoch=0073-step=007770.ckpt