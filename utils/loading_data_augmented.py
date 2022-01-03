from pathlib import Path
import pickle

import numpy as np

def get_IDs():
    dictionary = {}
    dictionary['train'] = []
    dictionary['valid'] = []
    list_files_path = '../../Datasets/300W_AFLW_Augmented/3dmm_data/train_aug_120x120.list.train'
    list_files = Path(list_files_path).read_text().strip().split('\n')
    dictionary['train'] = list_files
    list_files_path = '../../Datasets/300W_AFLW_Augmented/aflw2000_data/AFLW2000-3D_crop.list'
    list_files = Path(list_files_path).read_text().strip().split('\n')
    dictionary['valid'] = list_files
    return dictionary

def get_labels():
    list_files_path = '../../Datasets/300W_AFLW_Augmented/3dmm_data/train_aug_120x120.list.train'
    list_files = Path(list_files_path).read_text().strip().split('\n')
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