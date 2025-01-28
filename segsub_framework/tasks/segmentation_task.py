from enum import Enum
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from sam_segment import predict_masks_with_sam_prompts
from stable_diffusion_inpaint import fill_img_with_sd
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
lama_config = './lama/configs/prediction/default.yaml'
lama_ckpt = './pretrained_models/big-lama'

class Category(str, Enum):
    COLOR = 'color'
    SHAPE = 'shape'
    YESNO = 'yesno'
    NUMBER = 'number'
    
class SegmentationTask:
    def __init__(self, data_dir, category_label_dict):
        self.data = self.get_data(data_dir)
        self.data = self.filter_data()
        self.adversarial_samples = None
        assert all([Category(category) in Category for category in category_label_dict])
        self.category_label_dict = category_label_dict
    
    def get_data(self, data_dir):
        raise NotImplementedError
    
    def filter_data(self):
        raise NotImplementedError

    def process_data(self):
        for row in tqdm(self.data):
            # TODO: for all images in the row
            samples = self.get_samples(row)
            
            segmented_samples = []
            for sample in samples:
                if sample.num_detections > 0:
                    segmented_samples.append(sample)

            adversarial_samples = self.get_adversarial_samples(segmented_samples)
            for (label, seg_sub_image, idx, generation_id) in adversarial_samples:
                self.add_adversarial_sample(samples[idx], label, seg_sub_image, generation_id)
    
    # should return a SegmentationSamplde object
    def get_samples(self, row):
        raise NotImplementedError
    
    def get_adversarial_samples(self, seg_samples):
        if len(seg_samples) == 0:
            return []
        category = seg_samples[0].get_question_category()
        try:
            if category == Category.YESNO:
                # if seg_samples[0].answer_label == 'yes':
                adversarial_samples = []
                for idx, sample in enumerate(seg_samples):
                    adversarial_samples.append(('<RET>', sample.remove_object(), idx, 0))
                return adversarial_samples
                    
                # elif seg_samples[0].answer_label == 'no':
                    # TODO: are there any cases where we can inpaint objects to make the answer 'yes'?
                    # return []
            # elif category == Category.NUMBER:
            #     new_label = max(0, int(seg_sample.answer_label) - seg_sample.num_detections)
            #     new_image = seg_sample.remove_object()
            #     return [(str(new_label), new_image)]
            if category in [Category.COLOR, Category.SHAPE]:
                category_labels = self.category_label_dict[category].copy()
                if seg_samples[0].answer_label in category_labels:
                    category_labels.remove(seg_samples[0].answer_label)
                rand_category_labels = np.random.choice(category_labels, 50)
                adversarial_samples = []
                for gen_id, label in enumerate(rand_category_labels):
                    for idx, sample in enumerate(seg_samples):
                        seg_sub_image = sample.substitution(label)
                        adversarial_samples.append((label, seg_sub_image, idx, gen_id))
                return adversarial_samples
        except Exception as e:
            print(f"Error: {e}")
            return []

        raise ValueError(f"Invalid category: {category}")

    def add_adversarial_sample(self, original_sample, new_label, new_image, generation_id):
        raise NotImplementedError

    def quality_checks():
        raise NotImplementedError
        
        
        
class SegmentationSample():
    def __init__(self, question, label, image_path, question_category=None, infill_pipe=None, detector_pipe=None, segmenter=None, processor=None, image=None, segment_prompt=None):
        self.question = question
        if not segment_prompt:
            self.segment_prompt = self.get_question_object()
        else:
            self.segment_prompt = segment_prompt
        self.image_path = image_path
        if image is None:
            self.image = self.get_image(image_path)
        else:
            self.image = image
        self.mask = self.segment_image()
        self.answer_label = label
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

    def get_image(self, image_path):
        raise NotImplementedError    

    def get_question_object(self):
        raise NotImplementedError
    
    def get_question_category(self):
        raise NotImplementedError

    def segment_image(self):
        _, results = predict_masks_with_sam_prompts(
            Image.fromarray(self.image), 
            [self.segment_prompt],
            detector_pipe=self.detector_pipe,
            segmenter=self.segmenter
        )
        
        if results is None:
            # return zeros mask based on self.image.shape, but ignoring the RBG channel
            self.num_detections = 0
            return np.zeros_like(self.image)
        
        masks = [x.mask.astype(np.uint8) for x in results]
        masks = [dilate_mask(mask, 15) for mask in masks]
        # workaround for large # of objects (i.e. cv2.boundbox(combined_mask) is large)
        # masks = [masks[0]] 
        
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)

        for mask in masks:
            combined_mask = np.bitwise_or(combined_mask, mask)
        
        self.num_detections = len(masks)
        return combined_mask         

    def remove_object(self):
        if self.num_detections == 0:
            return self.image
        return inpaint_img_with_lama(self.image, self.mask, lama_config, lama_ckpt, device=device)

    def substitution(self, category_option, use_segment_prompt=True):
        if self.num_detections == 0:
            return self.image
        if use_segment_prompt:
            prompt = category_option + self.segment_prompt
        else:
            prompt = category_option
        return fill_img_with_sd(self.image, self.mask, prompt, device=device, pipe=self.infill_pipe)
