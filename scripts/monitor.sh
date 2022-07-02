#!/bin/bash
#apt-get install sysstat -y
#pip3 install gpustat
echo "time pcpu pmem gpu_usage gpu_mem total used free shared buff/cache available" > usage.csv
while true
do
  TIME=$(date +%T)
  CPU=$(ps -e -o pcpu,pmem,args --sort=pcpu | grep java | tail -n 1 | tr -s ' ' | cut -d" " -f1-3)
  GPU=$(gpustat --no-color | tail -n 1 | awk -F '[|,/]' '{ print $3 $4 }' | sed 's/%//g')
  MEM=$(free | tail -n 2 | head -n 1 | awk -F '[:]' '{ print $2 }')
  echo "$TIME $CPU $GPU $MEM" | tr -s ' ' >> usage.csv
  sleep 1
done
