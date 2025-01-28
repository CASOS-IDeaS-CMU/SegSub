import multiprocessing as mp
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from segment import *
import io
from PIL import Image
import os
import sys
sys.path.append('..')
import VoLTA.Multimodal_Coarse_Grained.write_webqa_multi as webqa
from importlib import reload
reload(webqa)
from functools import partial

import json 
import copy
import re
import time

# load pyarrow
import pyarrow as pa
import pandas as pd

import random
random.seed(0)

import spacy
nlp = spacy.load("en_core_web_sm")

def get_raw_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def convert_image_to_binary(image):
    with io.BytesIO() as output:
        # Save the image to the BytesIO object in JPEG format
        image.save(output, format='JPEG')
        # Get the binary data from the BytesIO object
        binary_data = output.getvalue()
    return binary_data

def get_first_noun(doc):
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text
    
    # return the longest word
    return max([token.text for token in doc], key=len)

def get_noun_phrase(doc):
   
    # Initialize variables
    object_noun = None
    
    # Iterate through the tokens to identify the object
    for token in doc:
        # Look for the direct object (dobj) which is a noun
        if token.dep_ == "dobj" and token.pos_ == "NOUN":
            object_noun = token
            break
        
        # Special handling for questions starting with "How many"
        if token.text.lower() in ["how", "many"] and token.dep_ in ["nummod", "attr"]:
            # Collect tokens that follow "how many"
            for child in token.children:
                if child.dep_ == "dobj" and child.pos_ == "NOUN":
                    object_noun = child
                    break
    
    # If we found a direct object, return its text
    if object_noun:
        return object_noun.text
    
    # If no direct object found, return the first noun chunk (usually the subject)
    noun_chunks = list(doc.noun_chunks)
    
    # Handle "What color" questions specifically
    if ' color ' in doc.text and len(noun_chunks) > 1:
        return noun_chunks[1].text
    
    if noun_chunks:
        return noun_chunks[-1].text.split()[-1]  # Return the last word of the last noun chunk

    return None

def extract_object(question):
   
    doc = nlp(question)
    obj = get_noun_phrase(doc)
    if obj:
        return obj
        
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            return token.text
    
    return get_first_noun(doc)

def perturb_vqav2_image(segment_prompt, inpaint_prompt, index, image_bytes, model, pipe):

    image_source, _, image_mask = get_frames_from_prompt("", segment_prompt, model, get_raw_image(image_bytes))
    # General perturbation: inpaint random replacement of same type of object
    # Note: sometimes this borks and just removes the object totally (particularly for small objects relative to rest of image)
    image_perturbed = inpaint_mask(inpaint_prompt, image_source, image_mask, pipe)
    return image_perturbed
    

def process_row(model, pipe, index):
    row = data.iloc[index]
    qa_list = list(zip(row['questions'], [x[0] for x in row['answers']]))
    perturbed_rows = []
    for i, (question, answer) in enumerate(qa_list):
        if answer.lower() in ['yes', 'no'] or question.lower().startswith(('what color', 'how many')):
            object_noun = extract_object(question)
            if answer.lower() in ['yes', 'no']:
                qcate = 'yesno'
                if 'yes' in answer.lower():
                    infill_prompt = "blank.png"
                    rand_answer = 'no'
                else:
                    infill_prompt = object_noun
                    rand_answer = 'yes'
            else:
                if question.lower().startswith('what color'):
                    qcate = 'color'
                else:
                    qcate = 'number'
                rand_answer = random.choice(webqa.domain_dict_gen[qcate])
                infill_prompt = rand_answer + ' ' + object_noun

            print(f"Q='{question}', A='{answer}', Infill='{infill_prompt}', Object='{object_noun}'")
            perturbed_image = perturb_vqav2_image(object_noun, infill_prompt, index, row['image'], model, pipe)
            new_row = copy.deepcopy(row)
            new_row['image'] = convert_image_to_binary(perturbed_image)
            new_row['answers'] = [rand_answer]
            new_row['questions'] = [question]
            perturbed_rows.append(new_row)
    print(f"Processed row {index}")
    return perturbed_rows


# Load data
vqa_path = "../data/VQAv2_arrows/vqav2_val.arrow"
data = pd.read_feather(vqa_path)
# data = data[:4]
schema = pa.Table.from_pandas(data).schema
batch_size = 90

# Define dataset root
dataset_root = vqa_path + ".perturbed"
os.makedirs(dataset_root, exist_ok=True)



def write_batches(writer, batch, schema):
    if batch:
        df_batch = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df_batch, schema=schema)
        writer.write_table(table)

if __name__ == '__main__': 
    
    mp.set_start_method('spawn')
    initialize_models()

    manager = mp.Manager()
    progress_queue = manager.Queue()
    pool = mp.Pool(processes=6)

    # Process data
    args = range(len(data))
    perturbed_data_batch = []
    
    # partially apply the process_row function with the model and pipe
    process_row_partial = partial(process_row, model, pipe)

    with pa.OSFile(f"{dataset_root}/rand_augmented.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, schema) as writer:
            results = pool.imap(process_row_partial, args)
            for result in results:
                perturbed_data_batch.extend(result)
                if len(perturbed_data_batch) >= batch_size:
                    write_batches(writer, perturbed_data_batch, schema)
                    perturbed_data_batch = []

            if perturbed_data_batch:
                write_batches(writer, perturbed_data_batch, schema)

    pool.close()
    pool.join()

    progress_queue.put(None)



# if __name__ == '__main__':
#     mp.set_start_method('spawn')

#     vqa_path = "../data/VQAv2_arrows/vqav2_val.arrow"
#     data = pd.read_feather(vqa_path)

#     manager = mp.Manager()
#     progress_queue = manager.Queue()
#     pool = mp.Pool(processes=2)

#     data = data[:4]  # For testing, adjust as needed
#     args = [(index, row, data) for index, row in data.iterrows()]

#     dataset_root = vqa_path + ".perturbed"
#     os.makedirs(dataset_root, exist_ok=True)

#     perturbed_data_batch = []
#     batch_size = 2  # Adjust batch size as needed

#     with pa.OSFile(f"{dataset_root}/rand_augmented.arrow", "wb") as sink:
#         with pa.RecordBatchFileWriter(sink, pa.Table.from_pandas(pd.DataFrame(columns=data.columns)).schema) as writer:
#             results = pool.imap(process_row, args)
#             for result in results:
#                 perturbed_data_batch.extend(result)
#                 if len(perturbed_data_batch) >= batch_size:
#                     write_batches(writer, perturbed_data_batch)
#                     perturbed_data_batch = []

#             if perturbed_data_batch:
#                 write_batches(writer, perturbed_data_batch)

#     pool.close()
#     pool.join()

#     perturbed_data_only = pd.DataFrame(perturbed_data_batch)
#     perturbed_table_only = pa.Table.from_pandas(perturbed_data_only)
#     with pa.OSFile(f"{dataset_root}/rand_only.arrow", "wb") as sink:
#         with pa.RecordBatchFileWriter(sink, perturbed_table_only.schema) as writer:
#             writer.write_table(perturbed_table_only)
    
# if __name__ == "__main__":
#     vqa_path = "../data/VQAv2_arrows/vqav2_val.arrow"
#     # load the VQA dataset
#     data = pd.read_feather(vqa_path)
#     data.head()
        
#     # new df with same schema as data that will be filled with perturbed 
#     vqa_path = "../data/VQAv2_arrows/vqav2_val.arrow"
#     data = pd.read_feather(vqa_path)
#     perturbed_data = copy.deepcopy(data)
#     dataset_root = vqa_path + ".perturbed"
#     os.makedirs(dataset_root, exist_ok=True)          

#     for index in tqdm(range(len(data)), desc="Perturbing images"):
#         qa_list = list(zip(data['questions'][index], [x[0] for x in data['answers'][index]]))
#         for i, (question, answer) in enumerate(qa_list):
#             if (answer.lower() in ['yes', 'no'] or question.lower().startswith(('what color', 'how many'))):
#                 object_noun = extract_object(question)
#                 if answer.lower() in ['yes', 'no']:
#                     qcate = 'yesno'
#                     if 'yes' in answer.lower():
#                         # remove and set answer to no
#                         infill_prompt = "blank.png"
#                         rand_answer = 'no'
#                     else:
#                         # add and set answer to yes
#                         infill_prompt = object_noun
#                         rand_answer = 'yes'
#                 else:
#                     # question.lower().startswith(('what color', 'how many')):
#                     if question.lower().startswith('what color'):
#                         qcate = 'color'
#                     else:
#                         qcate = 'number'      
#                     rand_answer = random.choice(webqa.domain_dict_gen[qcate])
#                     infill_prompt = rand_answer + ' ' + object_noun

#                 # print(question, answer, infill_prompt, object_noun)
#                 perturbed_image = perturb_vqav2_image(object_noun, infill_prompt, index, data['image'][index])
#                 new_row = copy.deepcopy(data.iloc[index])
#                 new_row['image'] = convert_image_to_binary(perturbed_image)
#                 new_row['answers'] = [rand_answer] 
#                 new_row['questions'] = [question]
#                 perturbed_data = pd.concat([perturbed_data, pd.DataFrame([new_row])], ignore_index=True)

#     train_table = pa.Table.from_pandas(perturbed_data)
    
#     # save perturbed_data as new pyarrow file 
#     with pa.OSFile(f"{dataset_root}/rand_augmented.arrow", "wb") as sink:
#             with pa.RecordBatchFileWriter(sink, train_table.schema) as writer:
#                 writer.write_table(train_table)

#     # remove original data
#     perturbed_data_only = perturbed_data.drop(data.index)
#     perturbed_table_only = pa.Table.from_pandas(perturbed_data_only)
#     with pa.OSFile(f"{dataset_root}/rand_only.arrow", "wb") as sink:        
#             with pa.RecordBatchFileWriter(sink, perturbed_table_only.schema) as writer:
#                 writer.write_table(perturbed_table_only)