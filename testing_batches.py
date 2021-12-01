import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.train import AdamOptimizer
from model.synergy import Synergy, create_synergy
from utils.data_utils.plotting_data import plot_landmarks_gt, plot_landmarks

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

input_shape = (450, 450, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
     "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
      "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

pose_3dmm = y[0]
exp_para = y[2]
shape_para = y[3]

pca = PCA(input_shape)
vertices_tf = pca(pose_3dmm, exp_para, shape_para)
vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
vertices = tf.make_ndarray(vertices)

for i in range(len(list_ids)):
    plot_landmarks(images[i], vertices[i], 'pred_'+str(i))
    plot_landmarks(images[i], y[1][i], name='gt_'+str(i))

model = Synergy(input_shape=input_shape)
model.model().summary()

model.save_weights("checkpoints/model_synergy.h5")
print()
print()
model_test = Synergy(input_shape=input_shape)
model_test.built = True
model_test.model()
model_test.load_weights("checkpoints/model_synergy.h5")
print(model_test.summary())