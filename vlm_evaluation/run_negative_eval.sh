# baseline
CUDA_VISIBLE_DEVICES=0 swift infer   --model_type qwen2-vl-7b-instruct   --model_id_or_path qwen/Qwen2-VL-7B-Instruct  \
    --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/qwen2

CUDA_VISIBLE_DEVICES=0 swift infer     --model_type phi3-vision-128k-instruct \
   --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/phi3

CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_5-7b-instruct \
    --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/llava15

# trained
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir ../output/negative_sampled/qwen2-vl-7b-instruct/v26-20241110-210452/checkpoint-2909 \
    --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/qwen2-ft

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir ../output/phi3-vision-128k-instruct/v18-20241111-125459/checkpoint-2909 \
    --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/phi3-ft

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir ../output/llava1_5-7b-instruct/v2-20241113-153540/checkpoint-2909 \
    --val_dataset ../data/val_shuffled.jsonl --dataset ../data/train.jsonl --result_dir ../output/negative_sampled/llava15-ft