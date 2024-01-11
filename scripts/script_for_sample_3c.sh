# Usage:
# bash script_for_sample_3c.sh CUR CUDA TOTAL CFG EXP PTH UC_FRAME UC_VIDEO UC_DOMAIN

CUR=$1
CUDA=$2
TOTAL=$3
CFG=$4
EXP=$5
PTH=$6
UC_FRAME=$7
UC_VIDEO=$8
UC_DOMAIN=$9

SEED=$[$CUR*111]

echo "Running the sampling script..."
echo "CUR="$CUR
echo "CUDA="$CUDA
echo "TOTAL="$TOTAL
echo "CFG="$CFG
echo "EXP="$EXP
echo "PTH="$PTH
echo "UC_FRAME="$UC_FRAME
echo "UC_VIDEO="$UC_VIDEO
echo "UC_DOMAIN="$UC_DOMAIN
echo "SEED="$SEED

source ~/.bashrc
conda activate stable
cd /public/mzsun/codes/GLOBER_PLUS/

CUDA_VISIBLE_DEVICES=$CUDA python generate_3c.py --base $CFG \
--resume experiments/$EXP/checkpoints/$PTH.ckpt \
--cur_part $CUR --total_part $TOTAL --seed $SEED \
--save_mode bybatch --video_length 16  --ngpu 1 \
--batch_size 8 --total_sample_number 64 --suffix "" \
--ucgs_frame $UC_FRAME --ucgs_video $UC_VIDEO --ucgs_domain $UC_DOMAIN

#--save_mode byframe --video_length 16 --total_sample_number 2048 \