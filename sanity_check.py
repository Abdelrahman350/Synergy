import cv2
import numpy as np
from data_generator.labels_preprocessing import denormalize, denormalize_DDFA
from model.synergy import Synergy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from losses import ParameterLoss, WingLoss
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from utils.loading_data import loading_generators, loading_test_examples
from tensorflow.keras.optimizers import Adam, Nadam
from model.morhaple_face_model import PCA, Reconstruct_Vertex
import os
from os import path

IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
model_path = "checkpoints/Synergy_DDFA_mse"
test = 'DDFA'
dataset = "DDFA"
morphable = 'DDFA' if test=='DDFA' else 'pca'
load_model = False

if not path.exists(f'checkpoints/'):
  os.makedirs(f'checkpoints/')
model_path = "checkpoints/" +"check_" + dataset
morphable = 'DDFA' if dataset=='DDFA' else 'pca'

list_ids, training_data_generator, validation_data_generator = loading_test_examples(dataset, input_shape)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)

model = Synergy(input_shape=input_shape, morphable=morphable)
optimizer = Adam(learning_rate=0.001)

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':0.02, 'Pm*':0.02, 'Lc':0.05, 'Lr':0.05}
model.compile(optimizer, losses, loss_weights=loss_weights)
if load_model:
   model.load_weights(model_path)

print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
   filepath=model_path,
   save_weights_only=True,
   monitor='val_Pm_loss',
   mode='min',
   save_best_only=True,
   verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_Pm_loss', factor=0.5, patience=5,\
   min_lr=0.00001, verbose=1)

print(f"\nThe training dataset has {len(training_data_generator.list_IDs)} training images.")
print(f"The validation dataset has {len(validation_data_generator.list_IDs)} validation images.\n")
model_fit = model.fit(
  x=images,
  y=y,
  epochs=100, 
  verbose=1,
  callbacks=[model_checkpoint_callback, reduce_lr])
print("Finished Training.")

DMM = model.predict(images)

poses_pred = DMM['Pm']

y_DMM = y['Pm']
poses_gt = y_DMM

pca = Reconstruct_Vertex(input_shape) if test=='DDFA' else PCA(input_shape)
vertices_tf = pca(poses_pred)
vertices_pred = vertices_tf.numpy()

vertices_tf = pca(y_DMM)
vertices_gt = vertices_tf.numpy()

lmks_output_path = 'check_output/landmarks/'
pose_output_path = 'check_output/poses/'
if not path.exists(f'check_output/'):
  os.makedirs(f'check_output/')
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
  if dataset == 'DDFA':
    pose_gt = denormalize_DDFA(poses_gt[i])
    pose_pred = denormalize_DDFA(poses_pred[i])
  else:
    pose_gt = denormalize(poses_gt[i])
    pose_pred = denormalize(poses_pred[i])

  gt = plot_pose(images[i], pose_gt, vertices_gt[i])
  pred = plot_pose(images[i], pose_pred, vertices_pred[i])
  comb = np.concatenate((gt, pred), axis=1)
  wfp = pose_output_path+list_ids[i].split('/')[-1]
  cv2.imwrite(wfp, comb)