import os
ROOT_DIR = os.getenv("ROOT_DIR")
import json
from tqdm import tqdm
import sys
sys.path.append("..")
from vlm_evaluation.eval_utils import *
import gc
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")
    parser.add_argument("--version", type=str, default="4")
    parser.add_argument("--perturbation_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--counterfactual", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)
    args = parser.parse_args()
    # model_path = "Qwen/Qwen2-VL-72B-Instruct-AWQ" #
    # model_path = "Qwen/Qwen2-VL-7B-Instruct" # 
    # model_path = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    model_path = args.model_path
    version = args.version
    perturbation_path = args.perturbation_path
    eval_data = json.load(open(args.eval_data_path, "r"))
    use_counterfactual_question = args.counterfactual == 1

    model, processor = get_model_processor(model_path)
    conversational_prompt = not 'Phi' in model_path
    output_str = "counterfactual" if use_counterfactual_question else "perturbation"
    output_file = f"data/qa_check_{output_str}_v{version}.csv"
    print(f"Output file: {output_file}")
    
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("model,file,qa_check\n")

    files = os.listdir(perturbation_path)    
    files = [file for file in files if int(file.split("_")[-1].split(".")[0]) >= 25]
    labels = json.load(open(f"{ROOT_DIR}/results/WebQA_train_val_obj_v2_generated_labels_shape_color.json", "r"))
    
    if args.end_idx > 0:
        files = files[args.start_idx:args.end_idx]
    
    for file in tqdm(files):
        if not file.endswith(".jpeg"):
            continue
        image_id = file.split("_")[0]
        key = file.split("_")[1].split(".")[0]
        example = eval_data[key]
        counterfactual_image_file = f"{perturbation_path}/{file}"
        images = None
        try:
            if use_counterfactual_question:
                question = f"Q: is the {example['Q_obj']} in both the original image and the perturbed image?"
                messages = get_messages(question, 2, ["original image", "perturbed image"], conversational_prompt)
                images = get_images([image_id, counterfactual_image_file])
            else:
                # question = f"Q: is the {example['Qcate']} of the {example['Q_obj']} the same in both images?"
                question = f"Q: what is the {example['Qcate']} of the {example['Q_obj']} in the image?"
                message = {"role": "user", "content": []}
                message["content"].append({"type": "image"})
                message["content"].append({"type": "text", "text": question})
                messages = [message]
                images = get_images([counterfactual_image_file])
            qa_check = run_inference(messages, images, processor, model, conversational_prompt)
        except Exception as e:
            # print(f"Error: {e}")
            torch.cuda.empty_cache()
            del messages
            if images:
                del images
            gc.collect()
            continue
        qa_check = qa_check.split('\n')[-1]

        if not use_counterfactual_question:
            # check for color match
            # imgs = labels.get(key, {}).get("img_posFacts", [])
            # imgs = [img for img_idx, img in enumerate(imgs) if img["image_id"] == int(image_id)]
            # if len(imgs) == 0:
            #     continue
            # labels[key]['A_perturbed'][]
            # imgs[0]
            # .get("label", "")
            try:
                perturbed_ans = labels[key]['A_perturbed'][file.split("_")[2].split(".")[0]]
                qa_check = 'No' if perturbed_ans in qa_check.lower() else 'Yes'
            except Exception as e:
                print(e)
                # continue

        with open(output_file, "a") as f:
            f.write(f"{model_path},{file},\"{qa_check}\"\n")

