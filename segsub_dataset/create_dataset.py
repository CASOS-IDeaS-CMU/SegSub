import os
ROOT_DIR = os.getenv("ROOT_DIR")
import json
import numpy as np
import pandas as pd
import os
import copy
import random

counterfactual_answer = '<RET> Sorry, I cannot determine the answer as there is not enough information. <RET>'
segsub_dir = f"{ROOT_DIR}/segsub_data/segsub_images"

domain_dict = {
    'color': ['orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'silver', 'chrome', 'green', 'aqua'],
    'shape': ['globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'],
    'yesno': ['yes', 'no'],
    'number': [str(i) for i in range(20,-1,-1)],
}

def file_passes_qa_check(file, qa_check_df):
    file = file.split('/')[-1]
    if not file in qa_check_df.index:
        return False
    return qa_check_df.loc[file]['qa_check'].strip().lower().startswith('no')


def get_conversation(entry, reverse_images = False, supervised = True):
    prompt = get_prompt(entry, reverse_images)
    conversations = [
        {
            "from": "human",
            "value": prompt
        },
    ]
    if supervised:
        conversations.append(
            {
                "from": "gpt",
                "value": entry['A'][0]
            }
        )
    return conversations

def get_prompt(data, reverse_images = False, img_token = "<image>"):
    imgs = data['img_posFacts'] if not reverse_images else data['img_posFacts'][::-1]
    prompt = ""
    for _, img in enumerate(imgs):
        prompt += f"{img_token}\n"
        if 'title' in img:
            prompt += f"Caption: {img['title']}\n"
    prompt += f"Q: {data['Q']}"
    return prompt
    

# def get_image_path(image_id):
#     # TODO: extract webqa images first
#     # return f"{img['image_id']}.jpeg")
#     pass

def convert_format(data, type, dataset):
    train_output = []
    val_output = []
    for key in data:
        entry = data[key]
        if 'img_posFacts' in entry and len(entry['img_posFacts']) > 0:
            conversations = get_conversation(entry)
            
            image_paths = []
            for img in entry['img_posFacts']:
                image_paths.append(img['image_id'])
            
            output_entry = {
                "system_prompt": "Answer the question Q. If you need help answer <RET> to get the context.",
                "image": image_paths,
                "conversations": conversations,
                "type": type,
                "dataset": dataset,
                "id": entry['id'],
                # "A": entry['A'],
            }
            
            if entry['split'] == 'train':
                train_output.append(output_entry)
            else:
                val_output.append(output_entry)
    return train_output, val_output

def substitute_label(ans, label, qcate = 'color'):
    category_labels = copy.deepcopy(domain_dict[qcate.lower()])
    category_labels.remove(label.lower())
    
    ans = ans.replace(',', ' ,').replace('.', ' .').replace('?', ' ?').replace('!', ' !')
    ans = ans.replace('(', ' (').replace(')', ' )').replace('[', ' [').replace(']', ' ]')
    ans = ans.replace('{', ' {').replace('}', ' }').replace(':', ' :').replace(';', ' ;')
    ans = ans.split(' ')
    
    for i, word in enumerate(ans):
        if word.lower() in category_labels:
            ans[i] = label
    # remove all labels from the answer
    # ans = [word for word in ans if word.lower() not in category_labels]
    
    ans = ' '.join(ans)
    ans = ans.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!')
    ans = ans.replace(' (', '(').replace(' )', ')').replace(' [', '[').replace(' ]', ']')
    ans = ans.replace(' {', '{').replace(' }', '}').replace(' :', ':').replace(' ;', ';')
    return ans

def get_perturbed_samples(train_data, perturbation_path, qa_check_df):
    data = copy.deepcopy(train_data)
    keys = list(data.keys())
    perturbed_samples = {}
    for k in keys:
        example = data[k]
        original_image_files = [str(img_data['image_id']) for img_data in example['img_posFacts']]
        for idx, label in example['A_perturbed'].items():
            idx = int(idx)
            # if samples_checked >= samples_per_image:
            #     break
            imgs = example['img_posFacts']
            if len(imgs) == 2 and idx % 2 == 1:
                continue

            try:
                generated_image_files = []
                for hack_idx, img in enumerate(imgs):
                    generated_file = f"{str(img['image_id'])}_{k}_{idx + hack_idx}.jpeg"
                    generated_path = os.path.join(perturbation_path, generated_file)
                    generated_image_files.append(generated_path)

                if any([not file_passes_qa_check(file, qa_check_df) for file in generated_image_files]):
                    continue
                
                perturbed_sample = copy.deepcopy(example)
                for img_idx, generated_image_path in enumerate(generated_image_files):
                    perturbed_sample['img_posFacts'][img_idx]['image_id'] = generated_image_path
                perturbed_sample['A'] = [substitute_label(perturbed_sample['A'][0], label, perturbed_sample['Qcate'])]
                perturbed_sample['id'] = k
                perturbed_samples[k + '_' + str(idx) + '_perturbed'] = perturbed_sample
                # samples_checked += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
    return perturbed_samples

def get_counterfactual_samples(train_data, perturbation_path, qa_check_df):
    data = copy.deepcopy(train_data)
    keys = list(data.keys())
    counterfactual_samples = {}
    for k in keys:
        example = data[k]
        generated_image_files = []

        try:
            for img in example['img_posFacts']:
                generated_file = f"{segsub_dir}/{perturbation_path}/{str(img['image_id'])}_{k}.jpeg"
                generated_image_files.append(generated_file)
            if any([not file_passes_qa_check(file, qa_check_df) for file in generated_image_files]):
                continue
            
            counterfactual_sample = copy.deepcopy(example)
            for img_idx, generated_image_path in enumerate(generated_image_files):
                counterfactual_sample['img_posFacts'][img_idx]['image_id'] = generated_image_path[len(segsub_dir):]
            counterfactual_sample['A'] = [counterfactual_answer]
            counterfactual_sample['id'] = k
            counterfactual_samples[k + '_counterfactual'] = counterfactual_sample
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    return counterfactual_samples

def get_conflicting_samples(train_data, perturbation_path, qa_check_df):
    data = copy.deepcopy(train_data)
    keys = list(data.keys())
    conflicting_samples = {}
    for k in keys:
        example = data[k]
        if len(example['img_posFacts']) != 2:
            continue
        for idx, label in example['A_perturbed'].items():
            idx = int(idx)
            imgs = example['img_posFacts']
            if len(imgs) == 2 and idx % 2 == 1:
                continue

            for img_idx, img in enumerate(imgs):
                try:
                    generated_file = f"{str(img['image_id'])}_{k}_{idx + img_idx}.jpeg"
                    passed_qa_check = file_passes_qa_check(generated_file, qa_check_df)
                    if not passed_qa_check:
                        continue
                    generated_path = os.path.join(perturbation_path, generated_file)                        
                    conflicting_sample = copy.deepcopy(example)
                    conflicting_sample['img_posFacts'][img_idx]['image_id'] = generated_path
                    conflicting_sample['A'] = ['<RET> Sorry, there is conflicting information. <RET>']
                    conflicting_sample['id'] = k
                    conflicting_samples[k + '_' + str(idx) + '_conflicting'] = conflicting_sample
                except Exception as e:
                    print(f"Error: {e}")
                continue
    return conflicting_samples


def get_okvqa_counterfactual_samples(split = 'val', perturbation_path = "okvqa/object_removal"):
    vqa_qid_obj_dir = f"{ROOT_DIR}/results/okvqa_val_obj.txt"
    split="val"
    val_questions_dir = f"{ROOT_DIR}/data/okvqa_{split}.json"
    # annotations can be downloaded here: https://okvqa.allenai.org/download.html
    val_annot_dir =  f"{ROOT_DIR}/data/mscoco_{split}2014_annotations.json"
    image_path = f"{ROOT_DIR}/data/coco-images"

    with open(val_questions_dir, 'r') as f:
        ques_data = json.load(f)

    with open(val_annot_dir, 'r') as f:
        annot_data = json.load(f)

    val_sample=1
    data = {}
    total = len(ques_data['questions'])
    max_sample = int(val_sample * total)
    for q in ques_data['questions'][:max_sample]:
        data[q['question_id']] = { "Q": q['question'], 'image_id': q['image_id']}

    for a in annot_data['annotations']:
        q_id = a['question_id']
        if q_id in data:
            data[q_id]["A"] = a['answers'][0]['raw_answer']

    def vqa_image_path(image_id):
        # format: val2014/COCO_val2014_000000543836.jpg
        return f"{image_path}/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg"

    eval_data = {}
    with open(vqa_qid_obj_dir, 'r') as f:
        for row in f:
            content = row.rstrip().split('\t')
            assert len(content) == 2
            qid = int(content[0])
            if qid not in data:
                continue
            llm_res = json.loads(content[1])
            eval_data[qid] = {
                'Q': data[qid]['Q'],
                'img_posFacts': [{'image_id': vqa_image_path(data[qid]['image_id'])}],
                'A': [data[qid]['A']],
                'qid': qid,
                'split': split,
                'img_id': data[qid]['image_id']
            }

    del data
    qa_check_df = pd.read_csv(f"{ROOT_DIR}/results/qa_check_okvqa_counterfactual_v2.csv")
    qa_check_df = qa_check_df.set_index('file')

    vqa_samples = {}
    vqa_counterfactual_samples = {}
    keys = list(eval_data.keys())
    for k in keys:
        example = eval_data[k]

        try:
            generated_file = f"{segsub_dir}/{perturbation_path}/{split}/{str(example['img_id'])}_{k}.jpeg"
            if not os.path.exists(generated_file) or not file_passes_qa_check(generated_file, qa_check_df):
                continue
            
            example['id'] = k
            vqa_samples[k] = copy.deepcopy(example)
            vqa_samples[k]['img_posFacts'][0]['image_id'] = vqa_samples[k]['img_posFacts'][0]['image_id'].split('/')[-1]
            example['img_posFacts'][0]['image_id'] = f"{perturbation_path}/{split}/{str(example['img_id'])}_{k}.jpeg"
            example['A'] = [counterfactual_answer]
            vqa_counterfactual_samples[str(k) + '_counterfactual'] = example
        except Exception as e:
            print(f"Error: {e}")
            continue

    return vqa_samples, vqa_counterfactual_samples

def get_vqa_counterfactual_samples(split = 'val', perturbation_path = "vqa/object_removal"):
    vqa_qid_obj_dir = f"{ROOT_DIR}/results/vqav2_{split}_obj.txt"
    
    # arrow format downloads for VQA based on prior work: https://github.com/LuoweiZhou/VLP#-misc
    vqa_data = pd.read_feather(f"{ROOT_DIR}/data/vqav2_{split}.arrow")
    image_path = f"{ROOT_DIR}/data/coco-images"
    qa_check_df = pd.read_csv(f"{ROOT_DIR}/results/qa_check_vqa_counterfactual_{split}.csv")
    qa_check_df = qa_check_df.set_index('file')

    def vqa_image_path(image_id):
        # format: val2014/COCO_val2014_000000543836.jpg
        return f"{image_path}/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg"

    eval_data = {}
    qid_img_q = {}
    for _,row in vqa_data.iterrows():
        img = row['image']
        for idx,qid in enumerate(row['question_id']):
            qid_img_q[str(qid)] = {"img": img, "q": row['questions'][idx], "img_id": row['image_id'], "answers": row['answers'][idx]}

    with open(vqa_qid_obj_dir, 'r') as f:
        for row in f:
            content = row.rstrip().split('\t')
            assert len(content) == 2
            qid = content[0]
            if qid not in qid_img_q:
                continue
            llm_res = json.loads(content[1])
            eval_data[qid] = {
                'Q': qid_img_q[qid]['q'],
                'img_posFacts': [{'image_id': vqa_image_path(qid_img_q[qid]['img_id'])}],
                'A': qid_img_q[qid]['answers'],
                'qid': qid,
                'split': split,
                'img_id': qid_img_q[qid]['img_id']
            }    
    
    vqa_samples = {}
    vqa_counterfactual_samples = {}
    keys = list(eval_data.keys())
    for k in keys:
        example = eval_data[k]

        try:
            generated_file = f"{segsub_dir}/{perturbation_path}/{split}/{str(example['img_id'])}_{k}.jpeg"
            if not os.path.exists(generated_file) or not file_passes_qa_check(generated_file, qa_check_df):
                continue
            
            example['id'] = k
            vqa_samples[k] = copy.deepcopy(example)
            vqa_samples[k]['img_posFacts'][0]['image_id'] = vqa_samples[k]['img_posFacts'][0]['image_id'].split('/')[-1]
            example['img_posFacts'][0]['image_id'] = f"{perturbation_path}/{split}/{str(example['img_id'])}_{k}.jpeg"
            example['A'] = [counterfactual_answer]
            vqa_counterfactual_samples[k + '_counterfactual'] = example
        except Exception as e:
            print(f"Error: {e}")
            continue

    return vqa_samples, vqa_counterfactual_samples


if __name__ == "__main__":
    random.seed(42)

    sample = None
    version = 4
    data = json.load(open(f"{ROOT_DIR}/data/WebQA_train_val.json", "r"))
    save = True
    # TODO: drop anything that doesn't have a QA check passing generation
    data = {k:v for k,v in data.items() if v['Qcate'].lower() in ['shape', 'color', 'yesno']}
    # perturbed_data = json.load(open("WebQA_train_val_obj_v2_generated_labels.json", "r"))
    perturbed_data = json.load(open(f"{ROOT_DIR}/results/WebQA_train_val_obj_v2_generated_labels_shape_color.json", "r"))
    perturbated_img_path = "webqa/object_perturbation"
    counterfactual_img_path = "webqa/object_removal"
    qa_check_perturbation_df = pd.read_csv(f"{ROOT_DIR}/results/qa_check_perturbation_v4.csv")
    qa_check_counterfactual_df = pd.read_csv(f"{ROOT_DIR}/results/qa_check_counterfactuals_v2.csv")
    qa_check_perturbation_df = qa_check_perturbation_df.set_index('file')
    qa_check_counterfactual_df = qa_check_counterfactual_df.set_index('file')
    
    for k in data:
        data[k]['id'] = k
    
    if sample:
        train_output, val_output = [], []
    else: 
        train_output, val_output = convert_format(data, 'original', 'webqa')

    print("Original dataset: ", len(train_output), len(val_output))

    conflicting_train_output, conflicting_val_output = convert_format(
        get_conflicting_samples(perturbed_data, perturbated_img_path, qa_check_perturbation_df),
        'conflicting', 'webqa'
    )
    print("Conflicting samples: ", len(conflicting_train_output), len(conflicting_val_output))
    if sample:
        conflicting_val_output = random.sample(conflicting_val_output, sample)
        
    train_output.extend(conflicting_train_output)
    val_output.extend(conflicting_val_output)

    perturbed_train_output, perturbed_val_output = convert_format(
        get_perturbed_samples(perturbed_data, perturbated_img_path, qa_check_perturbation_df),
        'perturbed', 'webqa'
    )
    print("Perturbed samples: ", len(perturbed_train_output), len(perturbed_val_output))
    if sample:
        perturbed_val_output = random.sample(perturbed_val_output, sample)
    train_output.extend(perturbed_train_output)
    val_output.extend(perturbed_val_output)

    counterfactual_train_output, counterfactual_val_output = convert_format(
        get_counterfactual_samples(data, counterfactual_img_path, qa_check_counterfactual_df),
        'counterfactual', 'webqa'
    )
    print("Counterfactual samples: ", len(counterfactual_train_output), len(counterfactual_val_output))
    if sample:
        counterfactual_val_output = random.sample(counterfactual_val_output, sample)
    train_output.extend(counterfactual_train_output)
    val_output.extend(counterfactual_val_output)
        
    vqa_train_samples, vqa_train_counterfactual_samples = get_vqa_counterfactual_samples('train')
    vqa_val_samples, vqa_val_counterfactual_samples = get_vqa_counterfactual_samples('val')
    
    vqa_counterfactual_train_output, _ = convert_format(vqa_train_counterfactual_samples, 'counterfactual', 'vqa')
    _, vqa_counterfactual_val_output = convert_format(vqa_val_counterfactual_samples, 'counterfactual', 'vqa')
    print("VQA counterfactual samples: ", len(vqa_counterfactual_train_output), len(vqa_counterfactual_val_output))
    if sample:
        vqa_counterfactual_val_output = random.sample(vqa_counterfactual_val_output, sample)
    train_output.extend(vqa_counterfactual_train_output)
    val_output.extend(vqa_counterfactual_val_output)
    
    vqa_train_output, _ = convert_format(vqa_train_samples, 'original', 'vqa')
    _, vqa_val_output = convert_format(vqa_val_samples, 'original', 'vqa')
    print("VQA original samples: ", len(vqa_train_output), len(vqa_val_output))
    if not sample:
        train_output.extend(vqa_train_output)
        val_output.extend(vqa_val_output)
        
    okvqa_train_samples, okvqa_train_counterfactual_samples = get_okvqa_counterfactual_samples('train')
    okvqa_val_samples, okvqa_val_counterfactual_samples = get_okvqa_counterfactual_samples('val')
    
    okvqa_counterfactual_train_output, _ = convert_format(okvqa_train_counterfactual_samples, 'counterfactual', 'okvqa')
    _, okvqa_counterfactual_val_output = convert_format(okvqa_val_counterfactual_samples, 'counterfactual', 'okvqa')
    print("OKVQA counterfactual samples: ", len(okvqa_counterfactual_train_output), len(okvqa_counterfactual_val_output))
    if sample:
        okvqa_counterfactual_val_output = random.sample(okvqa_counterfactual_val_output, sample)
    train_output.extend(okvqa_counterfactual_train_output)
    val_output.extend(okvqa_counterfactual_val_output)
    
    okvqa_train_output, _ = convert_format(okvqa_train_samples, 'original', 'okvqa')
    _, okvqa_val_output = convert_format(okvqa_val_samples, 'original', 'okvqa')
    print("OKVQA original samples: ", len(okvqa_train_output), len(okvqa_val_output))
    if not sample:
        train_output.extend(okvqa_train_output)
        val_output.extend(okvqa_val_output)
    
    print("Total samples: ", len(train_output), len(val_output))

    if save:
        if sample:
            with open(f'{ROOT_DIR}/segsub_data/segsub_data_labelling_sample.json', 'w') as f:
                json.dump(val_output, f, indent=4)
        else:
            with open(f'{ROOT_DIR}/segsub_data/segsub_data_train_v{version}.json', 'w') as f:
                json.dump(train_output, f, indent=4)
        
            with open(f'{ROOT_DIR}/segsub_data/segsub_data_val_v{version}.json', 'w') as f:
                json.dump(val_output, f, indent=4)

    print("Conversion complete. Output saved.")
