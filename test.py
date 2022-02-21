from data_generator.image_preprocessing import normalize_image
from data_generator.labels_preprocessing import Param3D_to_Pose, denormalize_param, denormalize_DDFA
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy

from model.morhaple_face_model import PCA, Reconstruct_Vertex
import numpy as np
from utils.loading_data import loading_test_examples
import cv2
import os
from os import path

set_GPU()
IMG_H = 160
input_shape = (IMG_H, IMG_H, 3)
model_path = "checkpoints/Synergy"
test = '300W_AFLW'
dataset = "AFLW"
morphable = 'DDFA' if test=='DDFA' else 'PCA'

list_ids, training_data_generator, validation_data_generator = loading_test_examples(dataset, input_shape)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)

images_ori = []
dataset_path='../../Datasets/300W_AFLW_Augmented/' if\
   dataset=='DDFA' else '../../Datasets/300W_AFLW/'
for id in list_ids:
  image_path = dataset_path + id
  image = cv2.imread(image_path)
  image = normalize_image(image)
  images_ori.append(image)

model = Synergy(input_shape=input_shape, morphable=morphable)

print(model.summary())
model.load_weights(model_path)

DMM = model.predict(images)
param3DMM_pred = DMM['Pm']

param3DMM_gt = y['Pm']
vertices_gt = y['Lc']

pca = Reconstruct_Vertex(input_shape) if test=='DDFA' else PCA(input_shape)
vertices_tf = pca(param3DMM_pred)
vertices_pred = vertices_tf.numpy()

lmks_output_path = 'test_output/landmarks/'
poses_output_path = 'test_output/poses/'
if not path.exists(f'test_output/'):
  os.makedirs(f'test_output/')
if not path.exists(lmks_output_path):
  os.makedirs(lmks_output_path)
if not path.exists(poses_output_path):
  os.makedirs(poses_output_path)

for i in range(len(list_ids)):
  gt = plot_landmarks(images[i], vertices_gt[i])
  pred = plot_landmarks(images[i], vertices_pred[i])
  comb = np.concatenate((gt, pred), axis=1)
  wfp = lmks_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, comb)

for i in range(len(list_ids)):
  if dataset == 'DDFA':
    pose_gt = denormalize_DDFA(param3DMM_gt[i])
    pose_pred = denormalize_DDFA(param3DMM_pred[i])
  else:
    pose_gt = denormalize_param(param3DMM_gt[i])
    pose_pred = denormalize_param(param3DMM_pred[i])
  
  theta_gt = Param3D_to_Pose(pose_gt)
  theta_pred = Param3D_to_Pose(pose_pred)
  gt = plot_pose(images[i], theta_gt, vertices_gt[i])
  pred = plot_pose(images[i], theta_pred, vertices_pred[i])
  comb = np.concatenate((gt, pred), axis=1)
  wfp = poses_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, comb)