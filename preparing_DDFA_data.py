from pathlib import Path
from os.path import join
import pickle
from tkinter import image_names
from unittest import skip
import pandas as pd
import cv2
from numpy import cos, sin
import numpy as np
from data_generator.labels_preprocessing import param3DMM_to_pose
from utils.data_utils.plotting_data import plot_pose

def draw_axis(image_original, pitch, yaw, roll):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    image = image_original.copy()
    pitch, yaw, roll = pitch, -yaw, roll

    tdx = 60
    tdy = 60
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
    
    llength = 100
    size = llength * 0.5

    # X-Axis pointing to right, drawn in red.
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing downward, drawn in green. 
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis pointing out of the screen, drawn in blue.
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(image, (int(tdx), int(tdy)), (int(x1),int(y1)), (0,0,255), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.line(image, (int(tdx), int(tdy)), (int(x3),int(y3)), (255,0,0), 2)
    return image

def parsing_pkl(file):
    pca_dir = '3dmm_data/'
    return pickle.load(open(pca_dir+file, 'rb'))

def denormalize(parameters_3DMM):
    param_mean = parsing_pkl('param_whitening.pkl').get('param_mean')[:62]
    param_std = parsing_pkl('param_whitening.pkl').get('param_std')[:62]
    parameters_3DMM = parameters_3DMM*param_std + param_mean
    return parameters_3DMM

def plot_pose(image, label):
    label = denormalize(label)
    pitch, yaw, roll = param3DMM_to_pose(label[:12])
    image = draw_axis(image, pitch, yaw, roll)
    return image

def column_refractor(row):
    row['Shape_Para'] = np.array(row['param3DMM'][12:52], dtype=np.float)
    row['Exp_Para'] = np.array(row['param3DMM'][52:62], dtype=np.float)
    row['Pose'] = np.array(row['param3DMM'][0:12], dtype=np.float)
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

list_IDs = Path(filelist_path).read_text().strip().split('\n')[0:5]
list_labels = pickle.load(open(labellist_path, 'rb')).tolist()[0:5]

data = pd.DataFrame(list(zip(list_IDs, list_labels)), columns=['image_ID', 'param3DMM'])
data = data.apply(column_refractor, axis=1)

dictionary = {}

data.apply(to_dictionary, axis=1)

print(dictionary)

i = 0
img_id = 'train_aug_120x120/' + data.iloc[i]['image_ID']
img_path = join(dataset_path, img_id)

image = cv2.imread(img_path)
poses = np.concatenate((dictionary[img_id]['Pose'],\
     dictionary[img_id]['Shape_Para'], dictionary[img_id]['Exp_Para']), axis=-1)
image = plot_pose(image, poses)
cv2.imwrite('try.jpg', image)
