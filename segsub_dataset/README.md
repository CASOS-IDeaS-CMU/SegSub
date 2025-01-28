### Overview

#### Environment Setup
```
conda create -n quality-checks python=3.10.15
cd SegmentationSubstitution/segsub_dataset
python install -r requirements.txt
```

#### Dataset Construction 
The quality checks take a long time to on generated images (24 hours per 30000 generations on a single NVIDIA RTX A6000). It is recommended to use separate GPUs per task. Generations may also be chunked into separate

```
conda activate quality-checks
export CUDA_VISIBLE_DEVICES=0
nohup python quality_check_webqa.py &
export CUDA_VISIBLE_DEVICES=1
nohup python quality_check_vqa.py &
export CUDA_VISIBLE_DEVICES=2
nohup python quality_check_okvqa.py &
```

Output from the quality checks can be used to delete generations that failed the quality check before the dataset is finalized.
```
python remove_bad_generations.py
```

Finally, we construct the dataset by generating the perturbed images with the relevant questions. We also add samples from the original dataset to ensure that performance on the original VQA tasks does not deteriorate as a result of finetuning.
```
python create_dataset.py
```

To evaluate the constructed dataset, see the [sample notebooks](segsub_label_task.ipynb). These were used for labeling tasks to determine the quality of perturbed samples that pass the automated quality checks.