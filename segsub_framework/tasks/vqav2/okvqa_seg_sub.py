import os
ROOT_DIR = os.getenv("ROOT_DIR")
import sys
sys.path.append('..')
sys.path.append('../..')
import json 
import copy
import re
from itertools import islice
import time
import cv2
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from tasks.segmentation_task import SegmentationSample, SegmentationTask
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from PIL import Image
import io
from utils import load_img_to_array, save_array_to_img, format_img

# load pyarrow
import pandas as pd

split="val"
val_questions_dir = f"{ROOT_DIR}/data/okvqa_{split}.json"
image_path = f"{ROOT_DIR}/data/coco-images/{split}2014/COCO_{split}2014_"
output_path = f"{ROOT_DIR}/segsub_data/segsub_images/okvqa/object_removal"
# Pre-process data to create the VQAv2 task
# 1. Get all the data into one dict (q_id: {img: <img>, obj: <obj>, new_label: <new_label>, actual_label: <actual_label>)

vqa_qid_obj_dir = 'okvqa_val_obj.txt'


with open(val_questions_dir, 'r') as f:
    ques_data = json.load(f)

val_sample=1
data = {}
total = len(ques_data['questions'])
max_sample = int(val_sample * total)
for q in ques_data['questions'][:max_sample]:
    data[q['question_id']] = { "Q": q['question'], 'image_id': q['image_id']}

vqa_data = {}
with open(vqa_qid_obj_dir, 'r') as f:
    for row in f:
        content = row.rstrip().split('\t')
        assert len(content) == 2
        qid = int(content[0])
        if qid not in data:
            continue
        llm_res = json.loads(content[1])
        vqa_data[qid] = {
            'object': llm_res['object'],
            'q': data[qid]['Q'],
            'new_label': llm_res['new_answer'],
            'qid': qid,
            'img_id': data[qid]['image_id']
        }

print(len(vqa_data))
del data


# Lets define the Task
from tasks.segmentation_task import SegmentationSample, SegmentationTask
class OKVQASegmentationSample(SegmentationSample):
    def __init__(self, row, infill_pipe, detector_pipe, segmenter, processor):
        self.q_obj = row['object']
        self.question_id = row['qid']
        self.image_id = row['img_id']
        self.infill_pipe = None
        self.detector_pipe = None
        self.segmenter = None
        self.processor = None
        if infill_pipe:
            self.infill_pipe = infill_pipe
        if detector_pipe:
            self.detector_pipe = detector_pipe
        if segmenter:
            self.segmenter = segmenter 
        if processor:
            self.processor = processor
        super().__init__(row['q'], row['new_label'], row['img_id'], question_category=None, infill_pipe=infill_pipe, detector_pipe=detector_pipe, segmenter=segmenter, processor=processor)
    
    def get_image(self, img):
        path = image_path + str(self.image_id).zfill(12) + ".jpg"
        return np.array(format_img(Image.open(path)))
    
    def get_question_object(self):
        return self.q_obj 

class OKVQASegmentationTask(SegmentationTask):
    def __init__(self, data_dict, infill_pipe, detector_pipe, segmenter, processor):
        self.data = self.get_data(data_dict)
        self.infill_pipe = infill_pipe
        self.detector_pipe = detector_pipe
        self.segmenter = segmenter
        self.processor = processor
        self.adversarial_samples = None
    
    def get_data(self, data_dict):
        d = [v for v in data_dict.values()]
        return d
    
    # Returns a SegementationSample (TODO)
    def get_samples(self, row):
        return [OKVQASegmentationSample(row, self.infill_pipe, self.detector_pipe, self.segmenter, self.processor)]
    
    def add_adversarial_sample(self, vqa_sample, new_label, new_image, generation_id):
       save_array_to_img(new_image, f'{output_path}/{str(vqa_sample.image_id)}_{vqa_sample.question_id}.jpeg'
)
    
    def get_adversarial_samples(self, seg_samples):
        # We expect only 1 sample here for VQAv2
        # And we do not care about categories anymore
        if len(seg_samples) == 0:
            return []
        assert len(seg_samples) == 1
        seg_sample = seg_samples[0]
        seg_sub_image = seg_sample.substitution(category_option='')
        return [(seg_sample.answer_label, seg_sub_image, 0, 0)]
        
### Load all the pipelines and models at once
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading SD model...")
infill_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
print("Loading SAM model...")
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"

detector_pipe = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
segmenter = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
processor = AutoProcessor.from_pretrained(segmenter_id)

# Reduce samples
vqa_task =OKVQASegmentationTask(vqa_data, infill_pipe=infill_pipe, detector_pipe=detector_pipe, segmenter=segmenter, processor=processor)

vqa_task.process_data()



