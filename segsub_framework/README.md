# Segmentation Substitution Framework

Based on the previous work [see here](https://github.com/geekyutao/Inpaint-Anything), except this uses prompt based SAM model (from [grounding-dino](https://github.com/IDEA-Research/GroundingDINO/tree/main)) instead of coordinate based model.

### TODO 
* generate adversarial examples for:
    - nlvr2
    - inherit from tasks.segmentation_task.SegmentationTask
    - use tasks.webqa.webqa_segmentation_task.WebQASegmentationTask as a reference
* quality check generated examples
* integrate generated examples with VoLTA data loaders
* finetune VoLTA for each task with & without generated examples

### Setup
```bash
conda create -n segsub python=3.8
conda activate segsub
pip install -r requirements.txt
```

Alternatively, if package version mismatches occur, try manually installing:
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
python -m pip install -r lama/requirements.txt
```
Download the model checkpoints provided in [LaMa](./lama/README.md) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

### Run WebQA Generation
First, extract objects from WebQA questions. 

Then run generation.

```bash 
export PYTHONPATH=$(pwd)
python3 tasks/webqa/get_webqa_question_object.py
python3 tasks/webqa/webqa_segmentation_task.py 
```
For evaluation of generated examples, see the [eval notebook](tasks/webqa/webqa_generation_eval.ipynb).

### VQAv2 and OKVQA Generation
Likewise for VQAv2:
```bash 
conda create -n gemini python=3.9
pip install requirements_gemini.txt
python3 tasks/vqa/vqav2_label_gen.py
python3 tasks/vqa/vqav2_seg_sub.py
```

And OKVQA:
```bash
python3 tasks/vqa/okvqa_label_gen.py 
python3 tasks/vqa/okvqa_seg_sub.py
```

### Adversarial Generation Method

This logic is encoded in the [SegmentationTask](tasks/segmentation_task.py) class.

| QCate | Method | New Answer | Quality Check |
| ----- | ------ | ---------- | ------------- |
| YesNo (Yes) | object removal | No / Unknown | No object detected |
| YesNo (No) | ? | ? | ? |
| Number | object removal | old answer - # objects removed | Decrease object detection threshold? |
| Color | infill: rand(color) + object | rand(color) | How to tell if color changed? |
| Shape | infill: rand(shape) + object | rand(shape) | ^ |

#### Object removal

```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --prompt "dog"
    --dilate_kernel_size 15 \
    --output_dir ./results
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

#### Object infilling

```bash
python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --seg_prompt "dog"
    --fill_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
```

#### Known issues
* segmentation fails 
  - current workaround: mock an all zero infill mask
* resizing after infill fails when segmentation finds multiple objects
  - current workaround: don't resize image if resize fails
  - potential workarounds: only use the first object detected
  - infill each mask 1 by 1 (slow for large number of objects)
```
File "utils/mask_processing.py", line 116, in crop_for_filling_post
    image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
ValueError: could not broadcast input array from shape (512,512,3) into shape (470,512,3)
```
