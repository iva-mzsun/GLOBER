# cmd format:
# bash sample.sh $CUR $CUDA $TOTAL

CUR=$1
CUDA=$2
TOTAL=$3
# ===============================
# bash script_for_sample_3c.sh CUR CUDA TOTAL CFG EXP PTH UC_FRAME UC_VIDEO UC_DOMAIN

CFG="configs/webvid/webvid_ldmae_8sec_v1.yaml"
EXP="2023-11-27T23-19-57_webvid_ldmae_8sec_v1"
PTH="last"

UC_FRAME=3.0
UC_VIDEO=3.0
UC_DOMAIN=3.0

bash scripts/script_for_sample_3c.sh $CUR $CUDA $TOTAL $CFG $EXP $PTH $UC_FRAME $UC_VIDEO $UC_DOMAIN

