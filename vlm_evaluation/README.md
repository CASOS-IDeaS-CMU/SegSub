### Overview
SWIFT can run inference on the entire SegSub evaluation set in 4-8 hours on a single NVIDIA RTX A6000, depending on the model.
Of course, evaluation across multiple models takes time. Consider running evaluation for each model in parallel, one model per available GPU.

#### VLM Evaluation Steps
```
conda activate quality-checks
```

For evaluating both baseline and finetuned models on the SegSub dataset:
```
python3 eval.py
```

<!-- ```
export CUDA_VISIBLE_DEVICES=0
./run_swift_eval ckpt_dir model_id
...

python3 eval.py
``` -->

For evaluating both baseline and finetuned models on randomly sampled counterfactual (query,image) pairs:
```
python3 ../vlm_finetuning/create_ms_swift_dataset.py --randomize True
./run_negative_eval.sh
```
