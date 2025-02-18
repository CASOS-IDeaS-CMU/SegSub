## KOALA :koala:: Knowledge Conflict Augmentations for Robustness in Vision Language Models 

1. [Introduction](#introduction)
2. [Repository Organization](#repository-organization)
3. [Dataset Setup](#koala-data)
4. [Reproduction of Results](#reproduction-of-results)
5. [License](#license)
<!-- 5. [Environment Setup](#environment-setup) -->


### Introduction

This repository accompanies the paper "KOALA :koala:: Knowledge Conflict Augmentations for Robustness in Vision Language Models". It provides the codebase, dataset, and supplementary materials used in our experiments. If you use, extend or build upon this project, please cite the following paper (under review at ACL 2025):
```
@article{carragher2025koala,
  title=KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models},
  author={Carragher, Peter and Jha Abhinand and R Raghav and Rao Nikitha and Carley, Kathleen M},
  journal={arXiv preprint arXiv:XXXX.XXXXXX},
  year={2025}
}
```
 Below, we describe the repository's structure and the purpose of each component.

### Repository Organization
```
KOALA/ 
├── augmentation/ 
├── quality_check/
├── vlm_finetuning/
├── vlm_evaluation/
├── figures/
```

- **`augmentation/`**: Framework implementation for generating knowledge conflicts in images (parametric, source, and counterfactual conflicts). Based upon [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything?tab=readme-ov-file).
- **`quality_check/`**: Tools for dataset construction, including automated VLM quality checks and notebooks for evaluating dataset samples manually.
- **`vlm_finetuning/`**: Code for fine-tuning Vision-Language Models using [SWIFT](https://github.com/modelscope/ms-swift).
- **`vlm_evaluation/`**: Evaluation scripts for both baseline and finetuned VLMs on both generated samples, as well as randomly sampled (query,image).
- **`figures/`**: Contains R scripts that generate the plots visualizations used in the paper.

### KOALA Data
Instead of having to generate a dataset from scratch, we provide the [KOALA dataset](https://www.doi.org/10.1184/R1/28297076), which was used for all experiments in the paper. Generating this dataset required some ~500 GPU hours. Note: this only includes images perturbed using the framework, and not the original samples. For finetuning and evaluation, the original VQA datasets must be downloaded in the [correct directory](#vqa-datasets).

Download the [KOALA dataset](https://www.doi.org/10.1184/R1/28297076) to KOALA/data and extract the generated images:

```
cd data
pip install py7zr
py7zr x images.7z
cd ..
```

#### Generation
Alternatively, perturbations can be generated locally. Note, this process takes some time and depends on both the number of VQA samples from the original datasets, and the number of generations per sample. GPU hour estimates for the 200,000 original generations are given (based on runtime with 2x NVIDIA RTX A6000).

1. download [VQA datasets](#vqa-datasets)
2. generate a similar dataset using the scripts in [augmentation](augmentation/README.md) (300 GPU hours), and 
3. use the [quality check scripts](quality_check/README.md) to filter out low quality generations (200 GPU hours).

#### VQA Datasets
In order to either generate an [alternative dataset](#generation) or finetune and evaluate VLMs, you will need to download the relevant VQA dataset.

The directory structure should be as follows:
```
KOALA/
├── data/  
│   ├── coco-images/  
│   │   ├── train2014/  
│   │   ├── val2014/  
│   ├── webqa-images/  
│   │   ├── imgs.lineidx  
│   │   ├── imgs.tsv  
│   ├── WebQA_train_val.json  
│   ├── vqav2_val.json  
│   ├── vqav2_train.json  
│   ├── okvqa_val.json  
│   ├── okvqa_train.json  
```

##### WebQA
Download and extract the WebQA image dataset from the [Google Drive](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ) provided by authors of [WebQA](https://github.com/WebQnA/WebQA) in `data/webqa-images`.
```
mkdir data/
mkdir data/webqa-images && cd data/webqa-images
py7zr x WebQA_imgs_7z_chunks/*.7z
py7zr x WebQA_data_first_release.7z
mv WebQA_train_val.json ../
cd ..
```

##### COCO Images
```
mkdir data/coco-images && cd data/coco-images/
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
```

##### VQAv2
```
cd data
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
mv v2_OpenEnded_mscoco_val2014_questions.json vqav2_val.json
mv v2_OpenEnded_mscoco_train2014_questions.json vqav2_train.json
cd ..
```

##### OK-VQA
```
cd data 
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip
unzip OpenEnded_mscoco_train2014_questions.json.zip
unzip OpenEnded_mscoco_val2014_questions.json.zip
mv OpenEnded_mscoco_val2014_questions.json okvqa_val.json
mv OpenEnded_mscoco_train2014_questions.json okvqa_train.json
cd ..
```

### Reproduction of Results
Based on these datasets, we can finetune and evaluate the robustness of VLMs to counterfactual samples and knowledge conflicts. GPU hour estimates for the training and evaluation sets are given (based on runtime with 2x NVIDIA RTX A6000).

For setting up and running finetuning using KOALA data and [SWIFT](https://github.com/modelscope/ms-swift), see our [finetuning scripts](vlm_finetuning/README.md) (16 GPU hours per model per epoch).

For setting up and running evaluation on the KOALA evaluation set (or any other dataset generated using the framework), see our [evaluation scripts](vlm_evaluation/README.md) (6 GPU hours per model).

Finally, run the [R scripts](figures/) to plot the results as shown in the paper.
<!-- ### Environment Setup

```
``` -->

