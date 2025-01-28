import json
import random
from PIL import Image

split = 'val'
dataset = json.load(open(f"../segsub_data/segsub_data_{split}_v4.json", "r"))
segsub_dir = "../segsub_data/segsub_images"
coco_image_dir = f"../data/coco-images/{split}2014"
webqa_image_dir = "../data/webqa-images"
randomize = True
sample = 50
random.seed(42)

def get_image_paths(sample):
    paths = []
    for image in sample['image']:
        if isinstance(image, int):
            assert(sample['dataset'] == 'webqa')
            paths.append(f"{webqa_image_dir}/{image}.jpeg")
        elif sample['type'] == 'original':
            assert(sample['dataset'] in ['vqa', 'okvqa'])
            paths.append(f"{coco_image_dir}/{image}")
        else:
            paths.append(f"{segsub_dir}/{image}")
    return paths

def convert_format(dataset):
    
    if randomize:
        nonperturbed_images = [sample for sample in dataset if sample['type'] == 'original' and len(sample['image']) == 1]
        dataset = [sample for sample in dataset if sample['type'] == 'counterfactual' and len(sample['image']) == 1]
    
    output = []
    for idx, sample in enumerate(dataset):
        if randomize:
            idx = random.randint(0, len(nonperturbed_images) - 1)
            image_paths = get_image_paths(nonperturbed_images[idx])
        else:
            image_paths = get_image_paths(sample)   
        output.append({
            "query": sample['conversations'][0]['value'],
            "response": sample['conversations'][1]['value'],
            "images": image_paths,
            "dataset": sample['dataset']
        })
    return output

dataset = convert_format(dataset)

if sample:
    with open(f"../data/{split}_shuffled_sample.jsonl", 'w') as f:
        datasets = ["webqa", "vqa", "okvqa"]
        
        prediction_file = "../output/negative_sampled/qwen2-ft.jsonl"
        prediction_data = []
        with open(prediction_file, "r") as f_preds:
            for line in f_preds:
                prediction_data.append(json.loads(line)["response"])
                
        for dataset_name in datasets:
            named_dataset = [(entry,prediction_data[idx]) for idx,entry in enumerate(dataset) if 'dataset' in entry and dataset_name == entry['dataset']]
            print(f"Number of {dataset_name} samples: {len(named_dataset)}")
            named_dataset = random.sample(named_dataset, sample)
            for entry,prediction in named_dataset:
                # save image
                image_path = entry['images'][0]
                image = Image.open(image_path)
                image_path = image_path.split('/')[-1]
                image_path = image_path.split('.')[0]
                entry['images'][0] = image_path + ".jpeg"
                entry["model_output"] = prediction
                output_path =f"../segsub_eval/sample_images/{entry['images'][0]}"
                image.save(output_path)
                json.dump(entry, f)

else:
    with open(f"../data/{split}_shuffled.jsonl", 'w') as f:
        for entry in dataset:
            json.dump(entry, f)
