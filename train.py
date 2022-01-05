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

input_shape = (450, 450, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='all',\
      input_shape=input_shape, batch_size=5, shuffle=True)

model = Synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.01)

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':0.02, 'Pm*':0.02, 'Lc':0.05, 'Lr':0.05}
model.compile(optimizer, losses, loss_weights=loss_weights)
print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
   filepath="checkpoints/model",
   save_weights_only=True,
   monitor='val_Pm_loss',
   mode='min',
   save_best_only=True,
   verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_Pm_loss', factor=0.1, patience=5,\
   min_lr=0.000001, verbose=1)

csv_logger = CSVLogger("checkpoints/training.csv", append=True)

print(f"\nThe training dataset has {len(training_data_generator.list_IDs)} training images.")
print(f"The validation dataset has {len(validation_data_generator.list_IDs)} validation images.\n")
model_fit = model.fit(
  x=training_data_generator,
  validation_data=validation_data_generator,
  epochs=200, 
  verbose=1,
  callbacks=[model_checkpoint_callback, reduce_lr, csv_logger])