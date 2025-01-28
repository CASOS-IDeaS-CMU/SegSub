### Overview

#### Environment Setup
```
conda create -n swift python=3.12
cd SegmentationSubstitution/vlm_finetuning
python3 install -r requirements.txt
```

To ensure system compatability, we recommend building flash-attn2 from source. Note: this process takes a long time. On a server with 20x Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz CPUs and 250Gb RAM, it takes approximately 18 hours.
In a more resource constrained environment, consider setting the max_jobs environment variable for stability (i.e. MAX_JOBS=4). Setting MAX_JOBS will result in a longer build time.
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install flash-attn --no-build-isolation
```

If building flash-attn2 is not feasible, you may attempt to use the prebuilt packages. Note: as of the time of writing, there are major incompatability issues between these packages and several different GPU architectures. Notably, flash-attn2 requires GPUs with the NVIDIA Amphere architecture or newer.

#### VLM Finetuning Steps
First, convert the SegSub dataset into a format that SWIFT accepts.
```
conda activate swift
python create_ms_swift_dataset.py
```

Next, run the training scripts to launch training jobs for the model of your choice. Note: training takes a long time on the SegSub dataset (12 hours per epoch on a 2x NVIDIA RTX A6000). It is recommended to use multiple GPUs per task, for which the `CUDA_VISIBLE_DEVICES` environment variable should be configured inside the scripts.
For finetuning Qwen2-VL, see [finetune_qwen.sh](finetune_qwen.sh). For Phi3-vision, see [finetune_phi.sh](finetune_phi.sh).