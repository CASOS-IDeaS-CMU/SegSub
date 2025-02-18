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

def get_raw_image(img):
    return Image.open(io.BytesIO(img)).convert("RGB")

vqa_qid_obj_dir = f"{ROOT_DIR}/results/vqav2_train_obj.txt"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument("--perturbation_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--counterfactual", type=int, default=1)
    args = parser.parse_args()
    # model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
    # model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
    # model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    model_path = args.model_path
    version = args.version
    perturbation_path = args.perturbation_path
    eval_data = pd.read_feather(args.eval_data_path)
    use_counterfactual_question = args.counterfactual == 1

    vqa_data = {}
    qid_img_q = {}
    for _,row in eval_data.iterrows():
        img = row['image']
        for idx,qid in enumerate(row['question_id']):
            qid_img_q[str(qid)] = {"img": img, "q": row['questions'][idx], "img_id": row['image_id']}


    with open(vqa_qid_obj_dir, 'r') as f:
        for row in f:
            content = row.rstrip().split('\t')
            assert len(content) == 2
            qid = content[0]
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


    model, processor = get_model_processor(model_path)
    conversational_prompt = not 'Phi' in model_path
    output_str = "counterfactual" if use_counterfactual_question else "perturbation"
    output_file = f"data/qa_check_{args.dataset}_{output_str}_v{version}.csv"
    print(f"Output file: {output_file}")
    
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("model,file,qa_check\n")

    for file in tqdm(os.listdir(perturbation_path)):
        if not file.endswith(".jpeg"):
            continue
        image_id = file.split("_")[0]
        key = file.split("_")[1].split(".")[0]
        example = vqa_data[key]
        counterfactual_image_file = f"{perturbation_path}/{file}"
        if use_counterfactual_question:
            question = f"Q: is the {example['object']} in both the original image and the perturbed image?"
        else:
            question = f"Q: is the {example['Qcate']} of the {example['Q_obj']} the same in both images?"

        messages = get_messages(question, 2, [], conversational_prompt)
        images = None
        # try:
        images = [get_raw_image(example['img'])]
        images.extend(get_images([counterfactual_image_file]))
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

