import json
from pathlib import Path
import pickle

import numpy as np

from data_generator.data_generator import DataGenerator
from utils.loading_data import loading_dictionaries

def get_IDs():
    dictionary = {}
    dictionary['train'] = []
    dictionary['valid'] = []
    list_files_path = '../../Datasets/300W_AFLW_Augmented/3dmm_data/train_aug_120x120.list.train'
    list_files = Path(list_files_path).read_text().strip().split('\n')
    list_files = ['train_aug_120x120/'+x for x in list_files]
    dictionary['train'] = list_files
    list_files_path = '../../Datasets/300W_AFLW_Augmented/aflw2000_data/AFLW2000-3D_crop.list'
    list_files = Path(list_files_path).read_text().strip().split('\n')
    list_files = ['aflw2000_data/'+x for x in list_files]
    dictionary['valid'] = list_files
    return dictionary

def get_labels():
    list_files_path = '../../Datasets/300W_AFLW_Augmented/3dmm_data/train_aug_120x120.list.train'
    list_files = Path(list_files_path).read_text().strip().split('\n')
    list_files = ['train_aug_120x120/'+x for x in list_files]
    labels_list = np.array(load('../../Datasets/300W_AFLW_Augmented/3dmm_data/param_all_norm_v201.pkl'))[:, :62]
    labels = dict(zip(list_files, labels_list))
    return labels
    
def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def load(fp):
    suffix = get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def loading_aug_dictionaries():
    base_dir_ids = base_dir_labels = '../../300W_AFLW_Augmented/'
    json_file_ids = base_dir_ids + 'IDs_augmented.json'
    json_file_labels = base_dir_labels + 'labels_augmented.json'

    with open(json_file_ids, 'r') as j:
        IDs = json.loads(j.read())
    with open(json_file_labels, 'r') as j:
        labels = json.loads(j.read())
    return IDs, labels

def loading_aug_generators(input_shape=(224, 224, 3), batch_size=16, shuffle=True):   
    partition_aug, labels_aug = loading_aug_dictionaries()
    training_data_generator = DataGenerator(partition_aug['train'], labels_aug,\
        batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)

    partition, labels = loading_dictionaries(dataset='AFLW')
    partition_combined = partition['train'] + partition['valid']
    validation_data_generator = DataGenerator(partition_combined, labels,\
        batch_size=batch_size, input_shape=input_shape, shuffle=shuffle)
    return training_data_generator, validation_data_generator