from metrics import OrientationMAE
from model.synergy import Synergy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from losses import ParameterLoss, WingLoss
from set_tensorflow_configs import set_GPU
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam, SGD
import os
from os import path

set_GPU()
dataset = '300W_AFLW'
param_loss = '_mse'
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
load_model = False
if not path.exists(f'checkpoints/'):
  os.makedirs(f'checkpoints/')
model_path = 'checkpoints/' +'Synergy'
morphable = 'DDFA' if dataset=='DDFA' else 'pca'

training_data_generator, validation_data_generator = loading_generators(dataset=dataset,\
      input_shape=input_shape, batch_size=64, shuffle=True)
training_data_generator.augmentation = True

model = Synergy(input_shape=input_shape, morphable=morphable)
optimizer = SGD(learning_rate=0.08, momentum=0.9, nesterov=True)

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='normal'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':0.02, 'Pm*':0.02, 'Lc':0.05, 'Lr':0.05}
metrics = {'Pm': [
  OrientationMAE(mode='pitch', name='pitch'), 
  OrientationMAE(mode='yaw', name='yaw'), 
  OrientationMAE(mode='roll', name='roll'), 
  OrientationMAE(mode='avg', name='avg')
  ]}
model.compile(optimizer, losses, loss_weights=loss_weights, metrics=metrics)

if load_model:
   model.load_weights(model_path)

print(model.summary())

model_checkpoint_tf = ModelCheckpoint(
  filepath=model_path,
  save_weights_only=True,
  monitor='val_loss',
  mode='min',
  save_best_only=True,
  verbose=1)

Pm_checkpoint_tf = ModelCheckpoint(
  filepath=model_path+'_Pm',
  save_weights_only=True,
  monitor='val_Pm_loss',
  mode='min',
  save_best_only=True,
  verbose=1)

model_checkpoint_h5 = ModelCheckpoint(
  filepath=model_path+'.h5',
  save_weights_only=True,
  monitor='val_loss',
  mode='min',
  save_best_only=True,
  verbose=0)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,\
   min_lr=0.00001, verbose=1)

csv_logger = CSVLogger(model_path+'.csv', append=False)

print(f"\nThe training dataset has {len(training_data_generator.list_IDs)} training images.")
print(f"The validation dataset has {len(validation_data_generator.list_IDs)} validation images.\n")
model_fit = model.fit(
  x=training_data_generator,
  validation_data=validation_data_generator,
  epochs=100, 
  verbose=1,
  callbacks=[model_checkpoint_tf, model_checkpoint_h5, reduce_lr, csv_logger, Pm_checkpoint_tf])
print("Finished Training.")