import time
import os
import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import json

val_questions_dir = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
val_annot_dir = 'data/v2_mscoco_train2014_annotations.json'
val_sample = 0.1
SYSTEM_PROMPT="""Answer in one word what is the object that this question is talking about. Do not provide the answer, just the object of the question.
Given the answer to the question, provide in one word an object that is completely different from the given answer and makes sense in the context of the question.
Make sure to answer in the following format without anything extra: 
{ "object" : <object_name>, "new_answer" : <new_answer> }

Consider this example as reference -
Q:“What is the mustache made of?”
A: "Banana"

{ "object" : "mustache", "new_answer" : "straw" }

Not generate a response for the input provided below."""


class BaseModel:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.api_token = self.get_api_token()
        self.system_prompt = system_prompt
        self.qualified_name = name

    def get_api_token(self):
        raise NotImplementedError

    def query(self, content):
        raise NotImplementedError

class GeminiModel(BaseModel):
    _generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
        
    def __init__(self, name: str, system_prompt: str):
        super().__init__(name=name, system_prompt=system_prompt)
        genai.configure(api_key=self.api_token)
        self.qualified_name="gemini-1.5-flash"
        self.model = genai.GenerativeModel(
            model_name=self.qualified_name,
            generation_config=self._generation_config,
            system_instruction=system_prompt
        )
    
    def get_api_token(self):
        api_token = os.getenv("GEMINI_API_TOKEN")
        if not api_token:
            raise ValueError("Please populate env variable 'GEMINI_API_TOKEN' with the access token")
        return api_token
    
    def query(self, content):
        response = self.model.generate_content(
            content,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            } 
        )
        return response.text

with open(val_questions_dir, 'r') as f:
    ques_data = json.load(f)

with open(val_annot_dir, 'r') as f:
    annot_data = json.load(f)

data = {}
total = len(ques_data['questions'])
max_sample = int(val_sample * total)
for q in ques_data['questions'][:max_sample]:
    data[q['question_id']] = { "Q": q['question']}

for a in annot_data['annotations']:
    q_id = a['question_id']
    if q_id in data:
        data[q_id]["A"] = a['multiple_choice_answer']

print("Num samples: ", len(data))

def get_remaining_ids(ans_dir):
    if not os.path.exists(ans_dir):
        return set()
    
    with open(ans_dir, 'r') as f:
        ans_ids = set()
        for line in f:
            ans_ids.add(line.split('\t')[0])

    remaining_ids = set()
    for id,d in data.items():
        if id not in ans_ids:
            remaining_ids.add(id)
    return remaining_ids

model = GeminiModel(name="flash", system_prompt=SYSTEM_PROMPT)
out_file = "okvqa_train_obj.txt"
remaining_ids = get_remaining_ids(out_file)
count = 0
for k,v in tqdm.tqdm(data.items()):
    if len(remaining_ids) > 0 and k not in remaining_ids:
        continue
    q = v['Q']
    a = v['A']
    prompt = f"Q: {q}\nA: {a}\n"
    try:
        resp = model.query(prompt)
    except Exception as e:
        continue
    finally:
        with open(out_file, 'a') as f:
            f.write(f"{k}\t{resp.strip()}\n")
