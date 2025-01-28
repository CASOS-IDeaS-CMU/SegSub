import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from eval_utils import *
import random

random.seed(42)

dataset = json.load(open("../data/segsub_data_val_v4.json", "r"))
segsub_dir = "../segsub_data/segsub_images"
split = "val"
coco_image_dir = f"../data/coco-images/{split}2014"

def get_image_paths(sample):
    paths = []
    for image in sample['image']:
        if isinstance(image, int):
            assert(sample['dataset'] == 'webqa')
            paths.append(image)
        elif sample['type'] == 'original':
            assert(sample['dataset'] in ['vqa', 'okvqa'])
            paths.append(f"{coco_image_dir}/{image}")
        else:
            paths.append(f"{segsub_dir}/{image}")
    return paths

original_samples = {}
for idx, sample in enumerate(dataset):
    if sample['type'] == 'original':
        sample['answer_idx'] = idx
        original_samples[f"{sample['dataset']}_{sample['id']}"] = sample

model_paths = [
    ("../finetuned_results/qwen2_finetuned.jsonl", "Qwen/Qwen2-VL-7B-Instruct"), # RET trained
    "Qwen/Qwen2-VL-7B-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "microsoft/Phi-3-vision-128k-instruct",
]

def eval_metrics(old_answer, new_answer, old_label, new_label, qcate=None, blank_answer=None): 
    res = {
        "img_old,label_old": ans_contains_correct_label(old_answer, old_label, qcate),
        "img_new,label_old": ans_contains_correct_label(new_answer, old_label, qcate),
        "img_old,label_new": ans_contains_correct_label(old_answer, new_label, qcate),
        "img_new,label_new": ans_contains_correct_label(new_answer, new_label, qcate),
        "img_old,label_ret": retrieval_predicted(old_answer),
        "img_new,label_ret": retrieval_predicted(new_answer),
    }
    if qcate and isinstance(qcate, list):
        res["img_old,label_any"] = ans_contains_any_label(old_answer, qcate)
        res["img_new,label_any"] = ans_contains_any_label(new_answer, qcate)
    if blank_answer:
        res["img_blank,label_old"] = ans_contains_correct_label(blank_answer, old_label, qcate)
        res["img_blank,label_new"] = ans_contains_correct_label(blank_answer, new_label, qcate) 
        res["img_blank,label_ret"] = retrieval_predicted(blank_answer)
    return res

results = {}
for model_path in model_paths:  
    if isinstance(model_path, tuple):
        model_path, original_model_id = model_path
    else:
        original_model_id = model_path
    
    if os.path.exists(f"../results/models/{model_path.split('/')[-1]}_results.json"):
        results[model_path] = json.load(open(f"../results/models/{model_path.split('/')[-1]}_results.json", "r"))
        continue    
    
    conversational_prompt = not 'Phi' in model_path
    
    if model_path.endswith(".jsonl"):
        model = None
        processor = None
        answers = []
        with open(model_path, "r") as f:
            for line in f:
                answers.append(json.loads(line))
    else:
        model, processor = get_model_processor(model_path, original_model_id)
    
    results[model_path] = {
        "webqa": {
            "perturbed": [],
            "conflicting": [],
            "counterfactual": [],
        },
        "vqa": {            
            # "perturbed": [],
            # "conflicting": [],
            "counterfactual": [],
        },
        "okvqa": {
            # "perturbed": [],
            # "conflicting": [],
            "counterfactual": [],
        },
    }
    
    # sample dataset
    dataset = random.sample(dataset, 500)
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        eval_dataset_id = sample['dataset']
        eval_type = sample['type']
        if eval_type == 'original' or not f"{eval_dataset_id}_{sample['id']}" in original_samples:
            continue
        original_sample = original_samples[f"{eval_dataset_id}_{sample['id']}"]
        old_label = original_sample['conversations'][1]["value"]

        new_label = sample['conversations'][1]["value"]
        if model:
            input_text = sample['conversations'][0]["value"]
            question = input_text.split("Q: ")[-1]
            captions = input_text.split("Q: ")[0].split("<image>\nCaption: ")
            if len(captions) > 1:
                captions = captions[1:]
                captions = [caption.split("\n")[0] for caption in captions]
            else:
                captions = []
            try:
                new_images = get_image_paths(sample)
                new_answer = eval_on_sample(new_images, question, captions, processor, model, conversational_prompt)
                old_images = get_image_paths(original_sample)
                old_answer = eval_on_sample(old_images, question, captions, processor, model, conversational_prompt)
            except Exception as e:
                print(f"Error: {e}")
                continue
        else:
            new_answer = answers[idx]['response']
            old_sample_idx = original_sample['answer_idx']
            old_answer = answers[old_sample_idx]['response']
    
        # TODO: blank image variants
        qcate = None
        if eval_dataset_id == 'vqa':
            qcate = 'vqa'
        elif eval_type == 'perturbed':
            qcate = ['color', 'shape']
        elif eval_type == 'conflicting':
            qcate = ['color', 'shape']
        elif eval_type == 'counterfactual':
            qcate = ['yesno']
        
        # TODO: Blank metrics
        results[model_path][eval_dataset_id][eval_type].append(eval_metrics(old_answer, new_answer, old_label, new_label, qcate))

    # Save results
    with open(f"../results/models/{model_path.split('/')[-1]}_results.json", "w") as f:
        json.dump(results[model_path], f)

# aggregate and save all results
for eval_dataset_id in ['webqa', 'vqa', 'okvqa']:
    for eval_type in ['perturbed', 'conflicting', 'counterfactual']:
        results_agg = {}
        for model_path in model_paths:
            if isinstance(model_path, tuple):
                model_path, _ = model_path
            results_agg[model_path] = {
                "img_old,label_old": [],
                "img_new,label_old": [],
                "img_old,label_new": [],
                "img_new,label_new": [],
                "img_old,label_ret": [],
                "img_new,label_ret": [],
            }
            for res in results[model_path][eval_dataset_id][eval_type]:
                for k, v in res.items():
                    results_agg[model_path][k].append(v)
            results_agg[model_path] = {k: np.mean(v) for k, v in results_agg[model_path].items()}
        # with open(f"../results/{eval_dataset_id}_{eval_type}_results.json", "w") as f:
        #     json.dump(results_agg, f)
        df = pd.DataFrame(results_agg).T
        df.to_csv(f"../results/{eval_dataset_id}_{eval_type}_results.csv")