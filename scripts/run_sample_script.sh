# cmd format:
# bash sample.sh $CUR $CUDA $TOTAL

# ps -ef | grep sample | awk '{print $2}' | xargs kill -9

for ((i=0;i<8;i++)) do
nohup bash scripts/sample.sh $[$i+1] $i 8 > logs/log$i.log 2>&1 &
done
