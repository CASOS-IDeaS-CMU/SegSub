import pandas as pd
from eval_utils import *
import os
from tqdm import tqdm

segsub_dir = "../segsub_data/segsub_images"

all_checks = [
    # (f"{segsub_dir}/vqa/object_removal/train/", '../data/qa_check_vqa_counterfactual_train.csv'),
    # (f"{segsub_dir}/vqa/object_removal/val/", '../data/qa_check_vqa_counterfactual_val.csv'),
    # (f"{segsub_dir}/webqa/object_removal/", '../data/qa_check_counterfactuals_v2.csv'),
    # (f"{segsub_dir}/webqa/object_perturbation/", '../data/qa_check_perturbation_v4.csv'),    
    (f"{segsub_dir}/okvqa/object_removal/val/", '../data/qa_check_okvqa_counterfactual_v2.csv'),
]

for generated_path, qa_check_file in all_checks:
    qa_check_df = pd.read_csv(qa_check_file)
    qa_check_df = qa_check_df.set_index('file')

    files = os.listdir(generated_path)
    for file in tqdm(files):
        if not file_passes_qa_check(file, qa_check_df):
            if os.path.exists(os.path.join(generated_path, file)):
                os.remove(os.path.join(generated_path, file))
                