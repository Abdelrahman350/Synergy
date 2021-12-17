import tensorflow as tf
from model.synergy import Synergy, create_synergy
from utils.data_utils.plotting_data import plot_landmarks, plot_pose

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model.morhaple_face_model import PCA
from utils.custom_fit import train, train_on_image
from losses import Synergy_Loss
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
optimizer = Nadam(learning_rate=0.1)
loss_function = tf.keras.losses.MeanSquaredError()

losses = {
  'output_1':loss_function,
  'output_2':loss_function,
  'output_3':loss_function,
  }
  
model.compile(optimizer, losses)
print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
   filepath="checkpoints/model.h5",
   save_weights_only=True,
   monitor='val_loss',
   mode='min',
   save_best_only=True,
   verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=2)


model_fit = model.fit(
  x=training_data_generator,
validation_data=validation_data_generator,
epochs=1000, 
verbose=1,
callbacks=[model_checkpoint_callback, reduce_lr])