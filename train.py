from metrics import OrientationMAE
from model.synergy import Synergy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping 
from losses import ParameterLoss, WingLoss
from set_tensorflow_configs import set_GPU
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam, SGD
import os
from os import path
from utils.plot_history import plot_history
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

set_GPU()
base_model_name = 'Synergy'
dataset = '300W_AFLW'
param_loss = 'MSE'
backbone = 'mobileNetV2'
model_name = backbone+'_'+param_loss

initial_lr = 0.08
momentum = 0.9
min_lr = 1e-4
patience_lr = 5
lr_reduce_factor = 0.5
batch_size = 64
patience = 20
optimizer = 'Nestrov'
epochs = 200

IMG_H = 160
input_shape = (IMG_H, IMG_H, 3)
load_model = False
model_path = os.path.join('checkpoints', base_model_name)

if not path.exists(f'checkpoints/'):
  os.makedirs(f'checkpoints/')
  os.makedirs(model_path)

morphable = 'DDFA' if dataset=='DDFA' else 'PCA'

training_data_generator, validation_data_generator = loading_generators(dataset=dataset,\
      input_shape=input_shape, batch_size=batch_size, shuffle=True)
training_data_generator.augmentation = True

model = Synergy(input_shape=input_shape, backbone=backbone)

if optimizer == 'Nestrov':
  optimizer = SGD(learning_rate=initial_lr, momentum=momentum, nesterov=True)
elif optimizer == 'Momentum':
  optimizer = SGD(learning_rate=initial_lr, momentum=momentum, nesterov=False)
elif optimizer == 'Adam':
  optimizer = Adam(learning_rate=initial_lr)
elif optimizer == 'Nadam':
  optimizer = Nadam(learning_rate=initial_lr)

print(f"\nThe algorithm used for optimization is {optimizer._name}")

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal', loss=param_loss),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='3dmm', loss=param_loss),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':0.02, 'Pm*':0.02, 'Lc':0.05, 'Lr':0.05}
metrics = {
  'Pm': [
    OrientationMAE(mode='pitch', name='pitch'), 
    OrientationMAE(mode='yaw', name='yaw'), 
    OrientationMAE(mode='roll', name='roll'), 
    OrientationMAE(mode='avg', name='avg')
    ]
  }
model.compile(optimizer, losses, loss_weights=loss_weights, metrics=metrics)

if load_model:
   model.load_weights(model_path)

print(model.summary())

model_checkpoint_tf = ModelCheckpoint(
  filepath=os.path.join(model_path, model_name),
  save_weights_only=True,
  monitor='val_loss',
  mode='min',
  save_best_only=True,
  verbose=1)

Pm_checkpoint_tf = ModelCheckpoint(
  filepath=os.path.join(model_path, model_name)+'_Pm',
  save_weights_only=True,
  monitor='val_Pm_loss',
  mode='min',
  save_best_only=True,
  verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_reduce_factor,\
   patience=patience_lr, min_lr=min_lr, verbose=1)

csv_logger = CSVLogger(os.path.join(model_path, model_name)+'.csv', append=False)

early_stop = EarlyStopping(monitor='val_loss', patience=patience)

print(f"\nThe training dataset has {len(training_data_generator.list_IDs)} training images.")
print(f"The validation dataset has {len(validation_data_generator.list_IDs)} validation images.\n")
model_fit = model.fit(
  x=training_data_generator,
  validation_data=validation_data_generator,
  epochs=epochs, 
  verbose=1,
  callbacks=[model_checkpoint_tf, Pm_checkpoint_tf, reduce_lr, csv_logger, early_stop])
print("Finished Training.")
plot_history(model_path, model_name)