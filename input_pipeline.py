from data_generator.labels_preprocessing import denormalize_param, denormalize_DDFA, param3DMM_to_pose
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from utils.loading_data import loading_test_examples
import cv2
import os
from os import path

dataset = 'AFLW'
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
morphable = 'DDFA' if dataset=='DDFA' else 'pca'

list_ids, training_data_generator, validation_data_generator = loading_test_examples(dataset, input_shape)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)

images_ori = []

for i in range(len(list_ids)):
  import numpy as np
  image = np.zeros((450, 450, 3))
  image[0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]
  images_ori.append(image)

y_DMM = y['Pm']
poses_gt = y_DMM
vertices = y['Lc']

lmks_output_path = 'inputPipeline_output/landmarks/'
pose_output_path = 'inputPipeline_output/poses/'
if not path.exists(f'inputPipeline_output/'):
  os.makedirs(f'inputPipeline_output/')
if not path.exists(lmks_output_path):
  os.makedirs(lmks_output_path)
if not path.exists(pose_output_path):
  os.makedirs(pose_output_path)


for i in range(len(list_ids)):
  image = plot_landmarks(images_ori[i], vertices[i])
  wfp = lmks_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, image)

for i in range(len(list_ids)):
  if dataset == 'DDFA':
    pose = denormalize_DDFA(poses_gt[i])
  else:
    pose = denormalize_param(poses_gt[i])
  
  theta = param3DMM_to_pose(pose)
  image = plot_pose(images_ori[i], theta, vertices[i])
  wfp = pose_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, image)