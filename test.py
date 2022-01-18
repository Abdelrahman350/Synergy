from data_generator.labels_preprocessing import denormalize, denormalize_DDFA
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy

from model.morhaple_face_model import PCA
import numpy as np
from utils.loading_data import loading_generators, loading_test_examples
import cv2
import os
from os import path

set_GPU()
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
model_path = "checkpoints/model"
test = "AFLW"

list_ids, training_data_generator, validation_data_generator = loading_test_examples(test, input_shape)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)

images_ori = []
dataset_path='../../Datasets/300W_AFLW/'
for id in list_ids:
  image_path = dataset_path + id
  image = cv2.imread(image_path)
  image = image.astype(float)
  image /= 255.0
  images_ori.append(image)

model = Synergy(input_shape=input_shape)

print(model.summary())
model.load_weights(model_path)

DMM = model.predict(images)
poses_pred = DMM['Pm']

y_DMM = y['Pm']
poses_gt = y_DMM

pca = PCA(input_shape)
vertices_tf = pca(poses_pred)
vertices_pred = vertices_tf.numpy()

vertices_gt = y['Lc']

lmks_output_path = 'inference_output/landmarks/'
pose_output_path = 'inference_output/poses/'
if not path.exists(f'inference_output/'):
  os.makedirs(f'inference_output/')
if not path.exists(lmks_output_path):
  os.makedirs(lmks_output_path)
if not path.exists(pose_output_path):
  os.makedirs(pose_output_path)

for i in range(len(list_ids)):
  gt = plot_landmarks(images[i], vertices_gt[i])
  pred = plot_landmarks(images[i], vertices_pred[i])
  comb = np.concatenate((gt, pred), axis=1)
  wfp = lmks_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, comb)

for i in range(len(list_ids)):
  gt = plot_pose(images[i], poses_gt[i], vertices_gt[i])
  pred = np.zeros_like(gt)
  try:
    pred = plot_pose(images[i], poses_pred[i], vertices_pred[i])
  except:
    print(f"\nPose prediction for image #{i}: {list_ids[i]} failed.")
    print("GT_param = ", poses_gt[i][:9])
    print("Pred_param = ", poses_pred[i][:9])
  comb = np.concatenate((gt, pred), axis=1)
  cv2.imwrite(f'output/test_{i}_poses.jpg', comb)

for i in range(len(list_ids)):
  if test == 'DDFA':
    pose_gt = denormalize_DDFA(poses_gt[i])
    pose_pred = denormalize_DDFA(poses_pred[i])
  else:
    pose_gt = denormalize(poses_gt[i])
    pose_pred = denormalize(poses_pred[i])

  gt = plot_pose(images[i], pose_gt, vertices_gt[i])
  pred = plot_pose(images[i], pose_pred, vertices_pred[i])
  comb = np.concatenate((gt, pred), axis=1)
  wfp = pose_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, image)