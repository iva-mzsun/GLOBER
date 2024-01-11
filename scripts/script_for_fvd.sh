# Usage:
# bash script_for_fvd.sh EXP UC PTH CUDA

EXP=$1
UC=$2
PTH=$3
CUDA=$4

echo "Running the fvd-test script..."
echo "EXP="$EXP
echo "UC="$UC
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
RESOLUTION=256
FAKE_PATH=experiments/$EXP/samples-byframe-ucgs$UC-$PTH-ddim50
#REAL_PATH=datasets/webvid/frames_1s/frames_16fps
#REAL_PATH=datasets/webvid/frames_1s_for_test/frames_16fps_2_cur0_total2
#REAL_PATH=datasets/SkyTimelapse/sky_timelapse/reorganized_frames/trainval/FPS32
REAL_PATH=datasets/ucf101/ucf101_fps32_test2048_wclasses
#REAL_PATH=datasets/ucf101/clips2048_union_256x

ROOT=/raid/mzsun/codes/StableBased/GLOBER_PLUS/
conda activate ldm
cd /raid/mzsun/codes/stylegan-v/

CUDA_VISIBLE_DEVICES=$CUDA python src/scripts/calc_metrics_for_dataset.py \
--real_data_path $ROOT/$REAL_PATH \
--fake_data_path $ROOT/$FAKE_PATH \
--mirror 1 --gpus 1 --resolution $RESOLUTION \
--metrics fvd2048_16f --verbose 1 --use_cache 0

