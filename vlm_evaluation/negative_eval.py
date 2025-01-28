import json
from pprint import pprint
import os

eval_files = [
    "../output/negative_sampled/qwen2-ft",
    "../output/negative_sampled/qwen2",
    "../output/negative_sampled/phi3-ft",
    "../output/negative_sampled/phi3",
    "../output/negative_sampled/llava15-ft",
    "../output/negative_sampled/llava15",
]

segsub_dataset = json.load(open(f"../segsub_data/segsub_data_val_v4.json", "r"))
nonperturbed_images = [sample for sample in segsub_dataset if sample['type'] == 'original' and len(sample['image']) == 1]
segsub_dataset = [sample for sample in segsub_dataset if sample['type'] == 'counterfactual' and len(sample['image']) == 1]


starting_phrases = [
    "<RET>",
    "Sorry", 
    "I cannot",# answer", 
    "I do not", 
]

retrieval_phrases = [
    " image does not", 
    " information", 
    " not enough", 
    " not clear", 
    " not visible", 
    " not sure", 
    " not able",
    " determine",
    " blurry",    
    " blurred",    
    " no existence",
    " context",
    " apologize",
    " sorry",
    "white background", 
]

def retrieval_predicted(answer, dataset):
    if dataset == "webqa":
        return int(any([phrase in answer for phrase in retrieval_phrases]) or any([phrase in answer for phrase in starting_phrases]))
    return int(any([phrase in answer for phrase in retrieval_phrases]) or any([phrase in answer for phrase in starting_phrases])) or answer == "no" or answer == "0"

all_results = {}
for folder in eval_files:
    # get first file in folder
    file = os.path.join(folder, os.listdir(folder)[0])
    with open(file, "r") as f:
        data = []
        for line in f:
            # print(len(data))
            data.append(json.loads(line))
        print(f"File: {file}, Number of entries: {len(data)}")
        
        res_dict = {
            "webqa": 0,
            "vqa": 0,
            "okvqa": 0,
        }
        for entry, (dataset, type) in zip(data, [(sample['dataset'], sample['type']) for sample in segsub_dataset]):
            if retrieval_predicted(entry["response"].strip(), dataset):#.startswith("<RET>"):
                # TODO: accept acknowledgments 
                res_dict[dataset] += 1
                
        for dataset in res_dict:
            res_dict[dataset] /= len([sample for sample in segsub_dataset if sample['dataset'] == dataset and sample['type'] == type])
        
        all_results[folder] = res_dict

# save to df
import pandas as pd
df = pd.DataFrame(all_results)
df = df.T
df.reset_index(inplace=True)
df['index'] = df['index'].apply(lambda x: x.split("/")[-1])
# rename index to model_path
df.rename(columns={"index": "model_path"}, inplace=True)
df.to_csv("negative_eval.csv", index=False)

    