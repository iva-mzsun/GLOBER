# Usage:
# bash scripts/script_for_fvd_3c.sh $EXP $UCFRAME $UCVID $UCDOMAIN $PTH $CUDA

EXP=$1
UCIMG=$2
UCVID=$3
PTH=$4
CUDA=$5

echo "Running the fvd-test script..."
echo "EXP="$EXP
echo "UCIMG="$UCIMG
echo "UCVID="$UCVID
echo "PTH="$PTH
echo "CUDA="$CUDA

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
conda activate ldm
cd /raid/mzsun/codes/stylegan-v/


RESOLUTION=256
ROOT=/raid/mzsun/codes/StableBased/GLOBER_PLUS/
FAKE_PATH=experiments/$EXP/samples-byframe-ucgsimg$UCIMG-ucgsvid$UCVID-$PTH
#REAL_PATH=datasets/SkyTimelapse/sky_timelapse/reorganized_frames/test_16f_2048videos_32fps
REAL_PATH=datasets/SkyTimelapse/sky_timelapse/reorganized_frames/trainval/FPS32
#REAL_PATH=datasets/ucf101/ucf101_fps32_test2048_wclasses

CUDA_VISIBLE_DEVICES=$CUDA python src/scripts/calc_metrics_for_dataset.py \
--real_data_path $ROOT/$REAL_PATH \
--fake_data_path $ROOT/$FAKE_PATH \
--mirror 1 --gpus 1 --resolution $RESOLUTION \
--metrics fvd2048_16f --verbose 1 --use_cache 0
