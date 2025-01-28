import json 
import os
ROOT_DIR = os.getenv("ROOT_DIR")
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print(os.getcwd())
import sys
# sys.path.append('..')
# # from segment import *

import tasks.webqa.arrow_format as webqa
from tasks.segmentation_task import SegmentationTask, SegmentationSample
from utils import load_img_to_array, save_array_to_img, format_img
import random
import numpy as np
random.seed(0)

webqa_category_labels = {
    #'color': ['yellow', 'blue', 'brown', 'gray', 'black', 'white', 'gold', 'pink', 'orange', 'purple', 'red', 'violet'],
    #'shape': ['ring', 'octagon', 'cross', 'pentagon', 'cone', 'ball', 'square', 'circle', 'star', 'triangle', 'sphere', 'hexagon'],
    'yesno': [],
    # 'number': [],
}

class WebQASegmentationTask(SegmentationTask):
    def __init__(self, data_dir):
        super().__init__(data_dir, webqa_category_labels)
        self.data_dir = data_dir

    def get_data(self, data_dir):
        with open(data_dir, 'r') as f:
            self.data_ = json.load(f)
            return self.data_.copy()

    def filter_data(self):
        keys = []
        for key in self.data.keys():
            content = self.data[key]
            qcate = content['Qcate'].lower()
            
            if not qcate in webqa_category_labels.keys():
                continue
            
            img_posFacts = content['img_posFacts']
            
            # TODO: for 2 imgs
            if not len(img_posFacts) in [1,2]:
                continue
    
            answer = content['A'][0]
            clean_answer = webqa.normalize_text(answer)
            answer_label = webqa.find_first_search_term(clean_answer, webqa.domain_dict[qcate], qcate, clean_answer)
            if answer_label is None or not answer_label in webqa.domain_dict[qcate]:
                continue
            keys.append(key)
        return [self.data[k] for k in keys]#[113:]
    
    def get_samples(self, row):
        if len(row['img_posFacts']) == 1:
            return [WebQASegmentationSample(row, 0)]
        else:
            return [WebQASegmentationSample(row, 0), WebQASegmentationSample(row, 1)]
    
    def add_adversarial_sample(self, webqa_sample, new_label, new_image, sample_id):
        output_path = f'{ROOT_DIR}/segsub_data/webqa/object_removal/{str(webqa_sample.image_id)}_{webqa_sample.question_id}_{str(sample_id)}.jpeg'
        save_array_to_img(new_image, output_path)
        if not 'A_perturbed' in self.data_[webqa_sample.question_id]:
            self.data_[webqa_sample.question_id]['A_perturbed'] = {}
        self.data_[webqa_sample.question_id]['A_perturbed'][sample_id] = new_label
        with open(self.data_dir.split('.')[0] + '_generated_labels.csv', 'a') as f:
            f.write(f"{webqa_sample.question_id},{webqa_sample.image_id},{sample_id},{new_label}\n")


class WebQASegmentationSample(SegmentationSample):
    def __init__(self, row, image_idx=0):
        self.question_category = row['Qcate'].lower()
        self.q_obj = row['Q_obj']
        self.question_id = row['Guid']
        self.image_id = row['img_posFacts'][image_idx]['image_id']
        image_path = self.image_id
        answer_label = WebQASegmentationSample.answer_2_label(row['A'][0], self.question_category)
        super().__init__(row['Q'], answer_label, image_path, self.question_category)
    
    @staticmethod
    def answer_2_label(answer, question_category):
        clean_answer = webqa.normalize_text(answer)
        return webqa.find_first_search_term(clean_answer, webqa.domain_dict[question_category], question_category, clean_answer)

    def get_question_object(self):
        # object_noun = extract_question_object(data[k]['Q'], data[k]['Qcate'])
        return self.q_obj # GPT object extraction 'beats' extract_question_object()
    
    def get_question_category(self):
        return self.question_category
    
    def get_image(self, image_id):
        img = webqa.read_image(image_id, webqa.lineidx)
        return np.array(format_img(img))
        
if __name__ == '__main__': 
    chunk_num = sys.argv[1] if len(sys.argv) > 1 else 'NA'

    if chunk_num == 'NA':
        data_path = f'{ROOT_DIR}/results/WebQA_train_val_obj_v2.json'
    else:
        data_path = f'{ROOT_DIR}/results/WebQA_train_val_obj_v2_chunk{chunk_num}.json'
    
    
    print(f'Processing: {data_path}')

    task = WebQASegmentationTask(data_path) # TODO: get train images
    task.process_data()
    data_path = data_path.split('.')[0]
    with open(f'{data_path}_perturbations_{chunk_num}.json', 'w') as f:
        json.dump(task.data_, f)

    
    # TODO: straight to pyarrow
    # arrow_path = "../data/multi_image_arrow_keyword/webqa_val.arrow"
    # arrow_data = pd.read_feather(arrow_path)
    # schema = pa.Table.from_pandas(arrow_data).schema

    # dataset_root = arrow_path + ".perturbed"
    # os.makedirs(dataset_root, exist_ok=True)

# logs
# chunk 0 -> gpu 6 - 3908451
# chunk 1 -> gpu 7 - 3908596
# chunk 2 -> gpu 8 - 
# chunk 3 -> gpu 9 - 

