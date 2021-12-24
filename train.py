import tensorflow as tf
from model.synergy import Synergy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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

input_shape = (120, 120, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=64, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
     "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
      "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

model = Synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.01)

losses = {
  'output_1': ParameterLoss(name='loss_Param_In', mode='normal'),
  'output_2': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'output_3': WingLoss(name='loss_LMK_f0'),
  'output_4': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'output_1':0.02, 'output_2':0.02, 'output_3':0.05, 'output_4':0.05}
model.compile(optimizer, losses, loss_weights=loss_weights)
print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
   filepath="checkpoints/model",
   save_weights_only=True,
   monitor='val_loss',
   mode='min',
   save_best_only=True,
   verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,\
   min_lr=0.00001, verbose=2)

model_fit = model.fit(
  x=training_data_generator,
  validation_data=validation_data_generator,
  epochs=1000, 
  verbose=1,
  callbacks=[model_checkpoint_callback, reduce_lr])