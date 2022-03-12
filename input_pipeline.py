from data_generator.labels_preprocessing import Param3D_to_Pose, denormalize_param, denormalize_DDFA
from model.synergy import Synergy
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
import cv2
import os
from os import path

from utils.loading_data import loading_generators

dataset = 'AFLW'
IMG_H = 160
input_shape = (IMG_H, IMG_H, 3)
morphable = 'DDFA' if dataset=='DDFA' else 'PCA'

training_data_generator, validation_data_generator, test_samples = loading_generators(dataset=dataset,\
      input_shape=input_shape, batch_size=1, shuffle=True)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(test_samples)

images_ori = []

for i in range(len(test_samples)):
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


for i in range(len(test_samples)):
  image = plot_landmarks(images_ori[i], vertices[i])
  wfp = lmks_output_path+test_samples[i].split('/')[-1]
  cv2.imwrite(wfp, image)

for i in range(len(test_samples)):
  if dataset == 'DDFA':
    pose = denormalize_DDFA(poses_gt[i])
  else:
    pose = denormalize_param(poses_gt[i])
  
  theta = Param3D_to_Pose(pose)
  image = plot_pose(images_ori[i], theta, vertices[i])
  wfp = pose_output_path+test_samples[i].split('/')[-1]
  cv2.imwrite(wfp, image)

# model = Synergy(input_shape=input_shape, morphable=morphable)
# model.predict(np.zeros((1, 128, 128, 3)))
# print(model.summary())
# model.save_weights('Synergy.h5')
# model.save_weights('Synergy')

# model_2 = Synergy(input_shape=input_shape, morphable=morphable)
# model_2.predict(np.zeros((1, 128, 128, 3)))
# model_2.load_weights('Synergy')
# model_3 = Synergy(input_shape=input_shape, morphable=morphable)
# model_3.predict(np.zeros((1, 128, 128, 3)))
# model_3.load_weights('Synergy.h5')