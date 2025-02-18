import os
import sys
sys.path.append('..')
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
import pandas as pd
from tasks.segmentation_task import SegmentationSample, SegmentationTask

class VQASegmentationSample(SegmentationSample):
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
        super().__init__(row['q'], row['new_label'], row['img'], question_category=None, infill_pipe=infill_pipe, detector_pipe=detector_pipe, segmenter=segmenter, processor=processor)
    
    def get_raw_image(self, img):
        return Image.open(io.BytesIO(img)).convert("RGB")

    def convert_image_to_binary(image):
        with io.BytesIO() as output:
            # Save the image to the BytesIO object in JPEG format
            image.save(output, format='JPEG')
            # Get the binary data from the BytesIO object
            binary_data = output.getvalue()
        return binary_data
    
    def get_image(self, img):
        raw_img = self.get_raw_image(img)
        return np.array(format_img(raw_img))
    
    def get_question_object(self):
        return self.q_obj 

class VQASegmentationTask(SegmentationTask):
    def __init__(self, data_dict, infill_pipe, detector_pipe, segmenter, processor):
        self.data = self.get_data(data_dict)
        self.infill_pipe = infill_pipe
        self.detector_pipe = detector_pipe
        self.segmenter = segmenter
        self.processor = processor
        self.adversarial_samples = None
    
    def get_data(self, data_dict):
        d = [v for v in data_dict.values()]
        d = d[len(d)//2:]
        return d
    
    # Returns a SegementationSample (TODO)
    def get_samples(self, row):
        return [VQASegmentationSample(row, self.infill_pipe, self.detector_pipe, self.segmenter, self.processor)]
    
    def add_adversarial_sample(self, vqa_sample, new_label, new_image, generation_id):
       output_path = f'{output_dir}/{str(vqa_sample.image_id)}_{vqa_sample.question_id}.jpeg'
       save_array_to_img(new_image, output_path)
    
    def get_adversarial_samples(self, seg_samples):
        # We expect only 1 sample here for VQAv2
        # And we do not care about categories anymore
        if len(seg_samples) == 0:
            return []
        assert len(seg_samples) == 1
        seg_sample = seg_samples[0]
        if seg_sample.num_detections == 0:
            return []
        seg_sub_image = seg_sample.remove_object()
        return [(seg_sample.answer_label, seg_sub_image, 0, 0)]

if __name__ == "__main__":
    # Replace with your VQAv2 path
    vqa_path = "{}/data/VQAv2_arrows/vqav2_train.arrow".format(os.path.expanduser("~/SegmentationSubstitution")) 
    # load the VQA dataset
    data = pd.read_feather(vqa_path)


    # Pre-process data to create the VQAv2 task
    # 1. Get all the data into one dict (q_id: {img: <img>, obj: <obj>, new_label: <new_label>, actual_label: <actual_label>)

    vqa_data = {}
    qid_img_q = {}
    vqa_qid_obj_dir = 'tasks/vqav2/vqav2_train_obj.txt'
    output_dir = "results/vqa_removal_train"

    for _,row in data.iterrows():
        img = row['image']
        for idx,qid in enumerate(row['question_id']):
            qid_img_q[qid] = {"img": img, "q": row['questions'][idx], "img_id": row['image_id']}


    with open(vqa_qid_obj_dir, 'r') as f:
        for row in f:
            content = row.rstrip().split('\t')
            assert len(content) == 2
            qid = int(content[0])
            if qid not in qid_img_q:
                continue
            llm_res = json.loads(content[1])
            vqa_data[qid] = {
                'object': llm_res['object'],
                'q': qid_img_q[qid]['q'],
                'img': qid_img_q[qid]['img'],
                'new_label': llm_res['new_answer'],
                'qid': qid,
                'img_id': qid_img_q[qid]['img_id']
            }

    print(len(vqa_data))
    del qid_img_q
    del data

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
    vqa_task = VQASegmentationTask(vqa_data, infill_pipe=infill_pipe, detector_pipe=detector_pipe, segmenter=segmenter, processor=processor)

    vqa_task.process_data()



