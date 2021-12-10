import json
import pandas as pd
from data_generator.labels_preprocessing import pose_to_3DMM
import numpy as np
import pickle
from utils.data_utils.label_parameters import *

path_to_dataset = '../../Datasets/'

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
        label['pose'] = get_pose_from_mat(path_to_dataset+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat(path_to_dataset+idx)
        label['Shape_Para'] = get_Shape_Para_from_mat(path_to_dataset+idx)
        labels[idx] = label
    print('Start Parsing valid files')
    for idx in dictionary['valid']:
        label = {}
        label['pose'] = get_pose_from_mat(path_to_dataset+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat(path_to_dataset+idx)
        label['Shape_Para'] = get_Shape_Para_from_mat(path_to_dataset+idx)
        labels[idx] = label
    return labels

def dictionary_to_json(dictionary, file_name):
    partition = json.dumps(dictionary, cls=NumpyEncoder)
    with open(file_name + '.json', 'w') as f:
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

def create_labels_json(data, dataset=['300W_LP']):
    IDs = get_IDs(data, dataset)
    dictionary_to_json(IDs, path_to_dataset+'IDs_'+dataset[0])
    labels = get_labels(IDs)
    dictionary_to_json(labels, path_to_dataset+'labels_'+dataset[0])
    data = pd.DataFrame(labels).transpose()
    data = data.apply(pose_to_param3DMM, axis=1)
    param3DMM_mean, param3DMM_std = get_mean_std(data, 'param3DMM')
    Exp_Para_mean, Exp_Para_std = get_mean_std(data, 'Exp_Para')
    Shape_Para_mean, Shape_Para_std = get_mean_std(data, 'Shape_Para')
    pose_mean, pose_std = get_mean_std(data, 'pose')
    param_62_mean = np.concatenate((param3DMM_mean, Exp_Para_mean, Shape_Para_mean), axis=0)
    param_62_std = np.concatenate((param3DMM_std, Exp_Para_std, Shape_Para_std), axis=0)
    dictionary = {'param_mean':param_62_mean, 'param_std':param_62_std}
    pickle.dump(dictionary, open(path_to_dataset+"param_"+dataset[0]+".pkl", "wb"))

def pose_to_param3DMM(row):
    row['param3DMM'] = pose_to_3DMM(row['pose'])
    return row

def get_mean_std(data, feature):
    mean = np.mean(data[feature].tolist(), axis=0)
    std = np.std(data[feature].tolist(), axis=0)
    return mean, std