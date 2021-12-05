import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.train import AdamOptimizer
from model.synergy import Synergy, create_synergy
from utils.data_utils.plotting_data import plot_landmarks, plot_pose

from model.morhaple_face_model import PCA
from utils.custom_fit import train
from losses import Synergy_Loss
from utils.loading_data import loading_generators
from model.synergy import create_synergy
from model.backbone import create_MobileNetV2

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

input_shape = (224, 224, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
     "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
      "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

model = Synergy(input_shape=input_shape)
model.model().summary()
model.build((1, input_shape[0], input_shape[1], input_shape[2]))
model.load_weights("checkpoints/Model.h5")

y_pred = model(images, training=False)
poses = y_pred[0].numpy()
vertices = y_pred[1].numpy()

for i in range(len(list_ids)):
  plot_landmarks(images[i], vertices[i], 'landmarks_pred_'+str(i))
  plot_landmarks(images[i], y[1][i], name='landmarks_gt_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images[i], poses[i], name='poses_pred_'+str(i))
  plot_pose(images[i], y[0][i], name='poses_gt_'+str(i))
