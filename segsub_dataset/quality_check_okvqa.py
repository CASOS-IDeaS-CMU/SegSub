import os
ROOT_DIR = os.getenv("ROOT_DIR")
import json
from tqdm import tqdm
import sys
from vlm_evaluation.eval_utils import *
import gc
import argparse
import pandas as pd
import io

def format_img(img):
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img

def get_vqa_image_path(img_id, image_path):
        return image_path + str(img_id).zfill(12) + ".jpg"

split="val"
val_questions_dir = f"{ROOT_DIR}/data/okvqa_val.json"
image_path = f"{ROOT_DIR}/data/coco-images/{split}2014/COCO_{split}2014_"
output_path = f"{ROOT_DIR}/segsub_data/segsub_images/okvqa/object_removal"
vqa_qid_obj_dir = f"{ROOT_DIR}/results/okvqa_val_obj.txt"


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument("--perturbation_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--counterfactual", type=int, default=1)
    args = parser.parse_args()
    # model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
    # model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
    # model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    model_path = args.model_path
    version = args.version
    perturbation_path = args.perturbation_path
    use_counterfactual_question = args.counterfactual == 1

    model, processor = get_model_processor(model_path)
    conversational_prompt = not 'Phi' in model_path
    output_str = "counterfactual" if use_counterfactual_question else "perturbation"
    output_file = f"../data/qa_check_{args.dataset}_{output_str}_v{version}.csv"
    print(f"Output file: {output_file}")
    
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("model,file,qa_check\n")

    for file in tqdm(os.listdir(perturbation_path)):
        if not file.endswith(".jpeg"):
            continue
        image_id = file.split("_")[0]
        key = int(file.split("_")[1].split(".")[0])
        if not key in vqa_data:
            print(f"Key {key} not in vqa_data")
            continue
        example = vqa_data[key]
        counterfactual_image_file = f"{perturbation_path}/{file}"
        if use_counterfactual_question:
            question = f"Q: is the {example['object']} in both the original image and the perturbed image?"
        else:
            question = f"Q: is the {example['Qcate']} of the {example['Q_obj']} the same in both images?"

        messages = get_messages(question, 2, [], conversational_prompt)
        # try:
        images = get_images([get_vqa_image_path(example['img_id'], image_path), counterfactual_image_file])
        qa_check = run_inference(messages, images, processor, model, conversational_prompt)
        # except Exception as e:
        #     # print(f"Error: {e}")
        #     torch.cuda.empty_cache()
        #     del messages
        #     if images:
        #         del images
        #     gc.collect()
        #     continue
        qa_check = qa_check.split('\n')[-1]
        with open(output_file, "a") as f:
            f.write(f"{model_path},{file},\"{qa_check}\"\n")

