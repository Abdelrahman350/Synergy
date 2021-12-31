import tensorflow as tf
from model.synergy import Synergy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from losses import ParameterLoss, WingLoss
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

input_shape = (128, 128, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=64, shuffle=True)

model = Synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.01)

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':1, 'Pm*':0.2, 'Lc':0.5, 'Lr':0.5}
model.compile(optimizer, losses, loss_weights=loss_weights)
print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
   filepath="checkpoints/model",
   save_weights_only=True,
   monitor='val_loss',
   mode='min',
   save_best_only=True,
   verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5,\
   min_lr=0.00001, verbose=1)

csv_logger = CSVLogger("checkpoints/training.csv", append=False)

model_fit = model.fit(
  x=training_data_generator,
  validation_data=validation_data_generator,
  epochs=2000, 
  verbose=1,
  callbacks=[model_checkpoint_callback, reduce_lr])