import json
from typing import Pattern
import numpy as np
from utils.data_utils.label_parameters import *

def get_IDs(data, list_datasets=['300W_LP', 'AFLW2000']):
    dictionary = {}
    dictionary['train'] = []
    dictionary['valid'] = []
    
    for index, row in data.iterrows():
        if row['type'] == 'train' and row['dataset'] in list_datasets:
            dictionary['train'].append(row['image'].split('.')[0])
        if row['type'] == 'val' and row['dataset'] in list_datasets:
            dictionary['valid'].append(row['image'].split('.')[0])
    return dictionary

def get_labels(dictionary):
    labels = {}
    print('Start Parsing train files')
    for idx in dictionary['train']:
        label = {}
        label['pose'] = get_pose_from_mat('Datasets/'+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat('Datasets/'+idx)
        label['Shape_Para'] = get_Shape_Para_from_mat('Datasets/'+idx)
        labels[idx] = label
    print('Start Parsing valid files')
    for idx in dictionary['valid']:
        label = {}
        label['pose'] = get_pose_from_mat('Datasets/'+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat('Datasets/'+idx)
        label['Shape_Para'] = get_Shape_Para_from_mat('Datasets/'+idx)
        labels[idx] = label
    return labels

def dictionary_to_json(dictionary, file_name):
    partition = json.dumps(dictionary, cls=NumpyEncoder)
    with open('Datasets/' + file_name + '.json', 'w') as f:
        f.write(partition + '\n')

def load_json(json_file_path):
    with open(json_file_path, 'r') as j:
        data = json.loads(j.read())
        j.close()
    return data

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_id_json(data):
    partition_LP = get_IDs(data, ['300W_LP'])
    dictionary_to_json(partition_LP, 'partition_LP')

    partition_AFLW = get_IDs(data, ['AFLW2000'])
    dictionary_to_json(partition_AFLW, 'partition_AFLW')
    return partition_LP, partition_AFLW

def create_labels_json(data):
    partition_LP, partition_AFLW = create_id_json(data)
    labels_LP = get_labels(partition_LP)
    dictionary_to_json(labels_LP, 'labels_LP')
    labels_AFLW = get_labels(partition_AFLW)
    dictionary_to_json(labels_AFLW, 'labels_AFLW')

