conda activate swift
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export SIZE_FACTOR=8
export MAX_PIXELS=602112
nohup swift sft \
  --model_type qwen2-vl-7b-instruct \
  --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
  --sft_type lora \
  --dataset data/train.jsonl \
  --val_dataset data/val.jsonl \
  --deepspeed default-zero2 \
  --num_train_epochs 1 --eval_steps 2909 &


CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-vl-7b-instruct/v26-20241110-210452/checkpoint-2909 \
    --load_dataset_config true --merge_lora true

# CUDA_VISIBLE_DEVICES=0 swift infer \
#   --model_type qwen2-vl-7b-instruct \
#   --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
#   --val_dataset data/val_shuffled.jsonl


# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --ckpt_dir ../output/qwen2-vl-7b-instruct/v26-20241110-210452/checkpoint-2909 \
#     --merge_lora true \
#     --dataset data/train.jsonl \
#     --val_dataset data/val.jsonl


# swift infer --model_type llava1_5-7b-instruct --dataset data/qwen_train.jsonl --val_dataset data/val.jsonl

# swift sft \
#   --model_type llava1_5-7b-instruct \
#   --sft_type lora \
#   --dataset data/train.jsonl \
#   --val_dataset data/val.jsonl \
#   --deepspeed default-zero2 \
#   --num_train_epochs 1 --eval_steps 2909

# CUDA_VISIBLE_DEVICES=0 swift infer \
#     --ckpt_dir output/llava1_5-7b-instruct/v2-20241113-153540/checkpoint-2909 \
#     --load_dataset_config true --merge_lora true
