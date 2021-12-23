import tensorflow as tf
from model.synergy import Synergy
from utils.data_utils.plotting_data import plot_landmarks

from model.morhaple_face_model import PCA
from utils.loading_data import loading_generators

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
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
     "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
      "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

DMM = y[0]
pose_3dmm = DMM[:, 0:12]
exp_para = DMM[:, 12:22]
shape_para = DMM[:, 22:62]

pca = PCA(input_shape)
vertices_tf = pca(DMM)
vertices = vertices_tf.numpy()

for i in range(len(list_ids)):
  plot_landmarks(images[i], vertices[i], 'pred_'+str(i))

model = Synergy(input_shape=input_shape)
model.summary()

model.save_weights("checkpoints/model_synergy.h5")
print()

model_test = Synergy(input_shape=input_shape)
model_test.build((1, input_shape[0], input_shape[1], input_shape[2]))
model_test.load_weights("checkpoints/model_synergy.h5")
print(model_test.summary())