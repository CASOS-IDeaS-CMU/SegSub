import json
import pandas as pd
# import pyarrow as pa
import random
import os
ROOT_DIR = os.getenv("ROOT_DIR")
# import pyarrow as ta
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
# from .glossary import normalize_word
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import base64
from word2number import w2n
import string, re
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])


data_path = f"{ROOT_DIR}/data/webqa-images"

with open(f"{data_path}/imgs.lineidx", "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]


# TODO choose category
domain_dict = {
    'color': ['orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'silver', 'chrome', 'green', 'aqua'],
    'shape': ['globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'],
    'yesno': ['yes', 'no'],
    'number': [str(i) for i in range(20,-1,-1)],
}
# all_domains = list(set(color_set + shape_set + yesno_set + written_numbers))

def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word

def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

def convert_image_to_binary(image):
    with BytesIO() as output:
        # Save the image to the BytesIO object in JPEG format
        image.save(output, format='JPEG')
        # Get the binary data from the BytesIO object
        binary_data = output.getvalue()
    return binary_data

def read_image(image_id, lineidx):
    with open(f"{data_path}/imgs.tsv", "r") as fp:
        fp.seek(lineidx[int(image_id)%10000000])
        imgid, img_base64 = fp.readline().strip().split('\t')
    assert int(image_id) == int(imgid), f'{image_id} {imgid}'
    im = Image.open(BytesIO(base64.b64decode(img_base64)))
    return im

def get_image(image_id, lineidx, resize = True):
    img = read_image(image_id, lineidx)
    if resize:
        img = img.resize((512,512))
    img = img.convert('RGB')
    return convert_image_to_binary(img)

def find_first_search_term(string, search_terms, qcate, clean_answer):
    answer_label = None
    for term in search_terms:
        if term in string:
            answer_label = term
    if not answer_label:
        if qcate == 'yesno':
            # Big dirty YES assumption hack
            # TODO: kill with fire, there is / there are / there are not, etc.
            answer_label = 'yes'
        elif qcate == 'number':
            if " no " in clean_answer or "none" in clean_answer:
                answer_label = '0'
            elif " once" in clean_answer or " single":
                answer_label = '1'
            elif "twice" in clean_answer:
                answer_label = '2'
            else:
                return None   
        else:
            return None
    return answer_label