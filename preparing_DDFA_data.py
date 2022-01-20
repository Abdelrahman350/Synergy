from pathlib import Path
from os.path import join
import pickle
import pandas as pd
import numpy as np
from utils.data_utils.data_preparing_utils import dictionary_to_json

def parsing_pkl(file):
    pca_dir = '3dmm_data/'
    return pickle.load(open(pca_dir+file, 'rb'))

def denormalize(parameters_3DMM):
    param_mean = parsing_pkl('param_whitening.pkl').get('param_mean')[:62]
    param_std = parsing_pkl('param_whitening.pkl').get('param_std')[:62]
    parameters_3DMM = parameters_3DMM*param_std + param_mean
    return parameters_3DMM

def column_refractor(row):
    row['Pose'] = np.array(row['param3DMM'][0:12], dtype=np.float32)
    row['Shape_Para'] = np.array(row['param3DMM'][12:52], dtype=np.float32)
    row['Exp_Para'] = np.array(row['param3DMM'][52:62], dtype=np.float32)
    return row

def to_dictionary(row):
    label = {}
    dir = 'train_aug_120x120/'
    label = {}
    label['Pose'] = row['Pose']
    label['Shape_Para'] = row['Shape_Para']
    label['Exp_Para'] = row['Exp_Para']
    dictionary[dir+row['image_ID']] = label

dataset_path = "../../Datasets/300W_AFLW_Augmented"
filelist = "3dmm_data/train_aug_120x120.list.train"
filelist_path = join(dataset_path, filelist)

labellist = "3dmm_data/param_all_norm_v201.pkl"
labellist_path = join(dataset_path, labellist)

list_IDs = Path(filelist_path).read_text().strip().split('\n')
json_IDs = ['train_aug_120x120/'+ x for x in list_IDs]
file_name = join(dataset_path, 'IDs_DDFA')
dictionary_to_json(json_IDs, file_name)

list_labels = pickle.load(open(labellist_path, 'rb')).tolist()

data = pd.DataFrame(list(zip(list_IDs, list_labels)), columns=['image_ID', 'param3DMM'])
data = data.apply(column_refractor, axis=1)

dictionary = {}

data.apply(to_dictionary, axis=1)
file_name = join(dataset_path, 'labels_DDFA')
dictionary_to_json(dictionary, file_name)