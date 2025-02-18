from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import pandas as pd
import tqdm
import json

# "Are the heads of Iranian women covered in traditional clothing?" Heads
# "What colors were the tent for the Ballard Chamber of Commerce Salmon Barbecue at Ballard Fest in 2006?" Tent
# "Does only one room in Graceland have light fixtures that hang from the ceiling?" Light Fixtures
# "Does the Osman Shah Mosque have fewer than three support pillars?" Pillars
# "How many arms does Joe Mays of the Broncos have tattooed?" Arms
# "Are the colors of the word lyric different in the Lyric Theater, Blacksburg and Lyric Theater, Georgia signs?" word lyric
# "How many holes does a Balalaika have in the body on the front of the instrument?" holes

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an NLP bot that extracts the object of questions, \
        such that an image segmentation system will be able to identify these objects in a photo of the scene. \
        If a question asks about the color or shape of an object, or how many of that object there are, return the name of that object. \
        'Color', 'Shape', or 'Number' are never valid answers and these words are not in your vocabulary."),
                # For yes/no questions, return both the object name and a description of the object that would negate the answer."),
    ("human", "What color flowers were displayed outside of the Hotel al codega san marco venezia in June of 2011?"),
    ("ai", "Flowers"),
    ("human", "What shape is the fountain near the arch in Washington Square Park?"),
    ("ai", "Fountain"),
    # ("human", "Does a Minnetonka Rhododendron flower have petals in a cup shape?"),
    # ("ai", "Petals"),    
    # ("human", "Are there sidewalks on both sides of the Mid-Hudson Bridge?"),
    # ("ai", "Sidewalks"),
    # ("human", "Does only one room in Graceland have light fixtures that hang from the ceiling?"),
    # ("ai", "Light Fixtures"),    
    ("human", "Do the awnings over the small windows have the same pattern as the ones over the bigger windows at the Washington Hotel?"),
    ("ai", "Windows"),
    ("human", "Is the Sumatran orangutan's skin on it's face as smooth as the skin on the Theropithecus gelada's face?"),
    ("ai", "Face"),
    # ("human", "What is the color of most lanterns at the entry for 2017 Taiwan Lantern Festival, Yunlin County, Taiwan?"),
    # ("ai", "Lanterns"),
    ("human", "{question}"),
])

chain = LLMChain(
    llm=ChatOpenAI(openai_api_key="<INSERT_KEY>", temperature=0.01, model="gpt-4o-mini"),
    prompt=chat_prompt,
)

objects = []
data = json.load(open('../../data/WebQA_train_val.json'))
output = '../../data/WebQA_train_val_obj_v2.json'
json.dump(data, open(output, 'w'))
for k in data:
    content = data[k]
    qcate = content['Qcate'].lower()
    
    # # TODO for train set too
    # if content['split'] != 'val':
    #     continue
    
    if not qcate in ['shape', 'color', 'number', 'yesno']:
        continue
    objects.append(chain.run(content['Q']))
    data[k]['Q_obj'] = objects[-1]
    print(f"{content['Q']}: {objects[-1]}")
json.dump(data, open(output, 'w'))
