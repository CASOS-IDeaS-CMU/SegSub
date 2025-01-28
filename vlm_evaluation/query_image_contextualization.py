import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
from openai import OpenAI
import requests
import base64
import os
ROOT_DIR = os.getenv("ROOT_DIR")

def is_valid(url):
    # only accept 'png', 'jpeg', 'gif', 'webp'
    if url[-4:] in ['.png', 'jpeg', '.gif', 'webp', '.jpg']:
        # check if the URL is live
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return False
        except:
            return False
        return True
    return False

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
def format_query(sample, image_path):
    query = sample['conversations'][0]['value']
    question = query.split("\n")[-1]
    file_type = image_path[-4:] == '.jpeg' and 'jpeg' or 'png'
    image_bytes = encode_image(image_path)
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/{file_type};base64,{image_bytes}"}},
        {"type": "text", "text": question},
    ]

def get_response(query):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Give a contextualization score for each image question pair. The score, between 1 and 10, should reflect the degree to which the image contextualizes the question. That is, how likely is it that you might come up with the question while looking at the image. Focus on the range of possible questions that might be asked about the image; that is, how likely is the given question, in this entire set. Give just the score, no explanation."},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(api_key="<INSERT_KEY>")

    MODEL="gpt-4o-mini"

    answers = {} 
    
    split = 'val'
    dataset = json.load(open(ff"{ROOT_DIR}/segsub_data/segsub_data_{split}_v4.json", "r"))
    segsub_dir = f"{ROOT_DIR}/segsub_data/segsub_images"
    coco_image_dir = ff"{ROOT_DIR}/data/coco-images/{split}2014"
    webqa_image_dir = f"{ROOT_DIR}/data/webqa-images/images"
    output_path = "segsub-gpt-4o-contextualization.json"
    
    def get_image_paths(sample):
        paths = []
        for image in sample['image']:
            if isinstance(image, int):
                assert(sample['dataset'] == 'webqa')
                assert(os.path.exists(f"{webqa_image_dir}/{image}.jpeg"))
                paths.append(f"{webqa_image_dir}/{image}.jpeg")
            elif sample['type'] == 'original':
                assert(sample['dataset'] in ['vqa', 'okvqa'])
                assert(os.path.exists(f"{coco_image_dir}/{image}"))
                paths.append(f"{coco_image_dir}/{image}")
            else:
                assert(os.path.exists(f"{segsub_dir}/{image}"))
                paths.append(f"{segsub_dir}/{image}")
        return paths

    output = []
    for sample in tqdm(dataset):
        if sample['type'] in ['perturbed', 'conflicting'] or len(sample['image']) != 1:
            output.append({
                "query": sample['conversations'][0]['value'],
                "images": image_paths,
            })
            with open(output_path, "a") as f:
                # write json to line
                f.write(json.dumps(output[-1]) + "\n")
            continue
        
        image_paths = get_image_paths(sample)   
        
        try:
            query = format_query(sample, image_paths[0])
            answer = get_response(query)
        except Exception as e:
            answer = str(e)

        output.append({
            "response": answer,
            "query": sample['conversations'][0]['value'],
            "images": image_paths,
        })

        with open(output_path, "a") as f:
            # write json to line
            f.write(json.dumps(output[-1]) + "\n")
            
        time.sleep(6)
