#!/bin/bash
#apt-get install sysstat -y
#pip3 install gpustat
echo "time  CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle  gpu_usage gpu_mem mem total used free shared  buff/cache   available" > usage.log
while true
do
  GPU=$(gpustat --no-color | tail -n 1 | awk -F '[|,]' '{ print $3 $4 }')
  CPU=$(mpstat | tail -n 1)
  MEM=$(free | tail -n 2 | head -n 1)
  echo $CPU $GPU $MEM >> usage.log
  sleep 1
done
