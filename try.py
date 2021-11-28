from losses import Synergy_Loss
from model.synergy import create_synergy
from model.morhaple_face_model import PCA
import numpy as np
from utils.data_utils.plotting_data import *
from data_generator.labels_preprocessing import *
import scipy.io as sio
from utils.loading_data import loading_generators
import tensorflow as tf

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


image, label = training_data_generator.get_one_instance('300W-LP/300W_LP/AFW/AFW_134212_1_2')
bfm_info = sio.loadmat('../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat')

pose_para = np.ravel(bfm_info['Pose_Para'])
pose_3dmm = np.ravel(pose_to_3DMM(pose_para))
shape_para = np.ravel(bfm_info['Shape_Para'][0:40])
exp_para = np.ravel(bfm_info['Exp_Para'][0:10])

pose_3dmm = np.expand_dims(pose_3dmm, 0)
shape_para = np.expand_dims(shape_para, 0)
exp_para = np.expand_dims(exp_para, 0)

pca = PCA(input_shape)
vertices_tf = pca(pose_3dmm, exp_para, shape_para)
vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
vertices = tf.make_ndarray(vertices)
#print(vertices.shape)
landmarks_pred = vertices

for output in landmarks_pred:
  plot_landmarks_pred(image, output, 'pred')
  plot_landmarks_gt(image, label, name='gt')

# model = create_synergy((224, 224, 3))
# print(model.summary())
