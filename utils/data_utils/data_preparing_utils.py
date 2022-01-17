import json
import pandas as pd
from data_generator.labels_preprocessing import normalize_dicts, pose_to_3DMM
import numpy as np
import pickle
from utils.data_utils.label_parameters import *
from pathlib import Path
from os.path import join

path_to_dataset = '../../Datasets/300W_AFLW/'

def get_IDs(data, list_datasets=['300W_LP', 'AFLW2000']):
    dictionary = {}
    dictionary['train'] = []
    dictionary['valid'] = []
    aflw_ids = get_AFLW2000_IDs()
    for index, row in data.iterrows():
        if 'AFLW2000' in list_datasets and row['image'] not in aflw_ids:
            continue
        if row['type'] == 'train' and row['dataset'] in list_datasets:
            dictionary['train'].append(row['image'])
        if row['type'] == 'val' and row['dataset'] in list_datasets:
            dictionary['valid'].append(row['image'])
    return dictionary

def get_labels(dictionary):
    labels = {}
    print('Start Parsing train files')
    for idx in dictionary['train']:
        label = {}
        label['Pose'] = pose_to_3DMM(get_pose_from_mat(path_to_dataset+idx))
        label['Shape_Para'] = get_Shape_Para_from_mat(path_to_dataset+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat(path_to_dataset+idx)
        labels[idx] = label
    print('Start Parsing valid files')
    for idx in dictionary['valid']:
        label = {}
        label['Pose'] = pose_to_3DMM(get_pose_from_mat(path_to_dataset+idx))
        label['Shape_Para'] = get_Shape_Para_from_mat(path_to_dataset+idx)
        label['Exp_Para'] = get_Exp_Para_from_mat(path_to_dataset+idx)
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
    data = pd.DataFrame(labels).transpose()
    Pose_mean, Pose_std = get_mean_std(data, 'Pose')
    Shape_Para_mean, Shape_Para_std = get_mean_std(data, 'Shape_Para')
    Exp_Para_mean, Exp_Para_std = get_mean_std(data, 'Exp_Para')
    param_62_mean = np.concatenate((Pose_mean, Shape_Para_mean, Exp_Para_mean), axis=0)
    param_62_std = np.concatenate((Pose_std, Shape_Para_std, Exp_Para_std), axis=0)
    dictionary = {'param_mean':param_62_mean, 'param_std':param_62_std}
    pickle.dump(dictionary, open("3dmm_data/"+"param_"+dataset[0]+".pkl", "wb"))
    labels = normalize_dicts(labels)
    dictionary_to_json(labels, path_to_dataset+'labels_'+dataset[0])

def pose_to_param3DMM(row):
    row['param3DMM'] = pose_to_3DMM(row['Pose'])
    return row

def get_mean_std(data, feature):
    mean = np.mean(data[feature].tolist(), axis=0)
    std = np.std(data[feature].tolist(), axis=0)
    return mean, std

def get_AFLW2000_IDs():
    dataset_path = "../../Datasets/300W_AFLW_Augmented"
    list_aflw_path = join(dataset_path, 'aflw2000_data/AFLW2000-3D_crop.list')
    list_aflw = Path(list_aflw_path).read_text().strip().split('\n')
    list_skip_path = join(dataset_path, 'aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy')
    list_skip = np.load(list_skip_path).tolist()
    aflw = []
    for i in range(len(list_aflw)):
        if i in list_skip:
            continue
        image_id = list_aflw[i]
        aflw.append('AFLW2000-3D/AFLW2000/'+image_id)
    return aflw