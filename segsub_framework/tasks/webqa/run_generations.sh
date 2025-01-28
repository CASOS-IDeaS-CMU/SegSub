#!/bin/bash
# sudo su
source segsub/bin/activate
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=$2
chunk_num=$1
nohup python tasks/webqa/webqa_segmentation_task.py $chunk_num > webqa_segmentation_task_yesno_$chunk_num.out 2>&1 &
watch "tail webqa_segmentation_task_yesno_$chunk_num.out"

#sudo ./tasks/webqa/run_generations.sh

