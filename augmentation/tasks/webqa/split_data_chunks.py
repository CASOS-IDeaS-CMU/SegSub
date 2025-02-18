import json
import os
import itertools

def split_data_chunks(data_path, num_chunks):
    with open(data_path, 'r') as f:
        data = json.load(f)
        print(f'total: {len(data)}')
    
    data = {key: data[key] for key in data if data[key]['Qcate']!='text'}
    print(f'after filter: {len(data)}')
    
    chunk_size = len(data)//num_chunks
    print(chunk_size*num_chunks)
    for i in range(num_chunks):
        print(f'split {i} | start: {i*chunk_size}, end: {(i+1)*chunk_size} path:{data_path.split(".")[0]}_chunk{i}.json')
        data_chunk = dict(itertools.islice(data.items(), i*chunk_size, (i+1)*chunk_size))
        
        with open(f"{data_path.split('.')[0]}_chunk{i}.json", 'w') as f:
            json.dump(data_chunk, f)
    
    # print(f'split {i+1} | start: {(i+1)*chunk_size}, end: {len(data)} path:{data_path.split(".")[0]}_chunk{i+1}.json')
    
    #data_chunk = {key:data[key] for key in list(data.keys())[(i+1)*chunk_size:]}
    
    #data_chunk = dict(itertools.islice(data.items(), (i+1)*chunk_size, len(data)))
    
    #with open(f'{data_path.split(".")[0]}_chunk{i+1}.json', 'w') as f:
    #    json.dump(data_chunk, f)

def get_data_distribution(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
        print(f'total: {len(data)}')
    
    data = {key: data[key] for key in data if data[key]['Qcate']!='text'}
    print(f'after filter: {len(data)}')
    
    shape_data = {key: data[key] for key in data if data[key]['Qcate']=='shape'}
    print(f'after filter shape_data: {len(shape_data)}')
        
    color_data = {key: data[key] for key in data if data[key]['Qcate']=='color'}
    print(f'after filter color_data: {len(color_data)}')
    
    yesno_data = {key: data[key] for key in data if data[key]['Qcate']=='yesno'}
    print(f'after filter yesno_data: {len(yesno_data)}')
    
    number_data = {key: data[key] for key in data if data[key]['Qcate']=='number'}
    print(f'after filter number_data: {len(number_data)}')
    

if __name__ == '__main__': 
    data_path = '{}/VQA_data/WebQA_train_val_obj_v2.json'.format(os.path.expanduser("~/SegmentationSubstitution"))
    
    
    #split_data_chunks(data_path, num_chunks=4)
    get_data_distribution(data_path)
    
