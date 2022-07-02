#!/bin/bash
#apt-get install sysstat -y
#pip3 install gpustat
echo "time  CPU  gpu_usage gpu_mem mem total used free shared  buff/cache   available" > usage.log
while true
do
  TIME=$(date +%T)
  CPU=$(ps -e -o pcpu,pmem,args --sort=pcpu | grep java | tail -n 1 | cut -d" " -f1-4)
  GPU=$(gpustat --no-color | tail -n 1 | awk -F '[|,/]' '{ print $3 $4 }' | sed 's/%//g')
  MEM=$(free | tail -n 2 | head -n 1 | awk -F '[:]' '{ print $2 }')
  echo "$TIME $CPU $GPU $MEM" | tr -s ' ' >> usage.log
  sleep 1
done
