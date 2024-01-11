# Usage:
# bash script_for_sample.sh CFG EXP PTH UC CUR TOTAL CUDA

CFG=$1
EXP=$2
PTH=$3
UC=$4
CUR=$5
TOTAL=$6
CUDA=$7
SEED=$[$7*111]

echo "Running the sampling script..."
echo "CFG="$CFG
echo "EXP="$EXP
echo "PTH="$PTH
echo "UC="$UC
echo "CUR="$CUR
echo "TOTAL="$TOTAL
echo "CUDA="$CUDA
echo "SEED="$SEED

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/raid/shchen/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/raid/shchen/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/raid/shchen/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/raid/shchen/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate stable
cd /raid/mzsun/codes/StableBased/GLOBER_PLUS/
CUDA_VISIBLE_DEVICES=$CUDA python generate.py --base $CFG --resume experiments/$EXP/checkpoints/$PTH.ckpt \
--save_mode byframe --video_length 16 --total_sample_number 2048 \
--unconditional_guidance_scale $UC --ngpu 1 --batch_size 8 \
--cur_part $CUR --total_part $TOTAL --seed $SEED