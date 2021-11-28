from losses import Synergy_Loss
from model.synergy import create_synergy
from model.morhaple_face_model import PCA
import numpy as np
from utils.data_utils.plotting_data import *
from data_generator.labels_preprocessing import *
from data_generator import data_generator
import scipy.io as sio

import json
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

json_file_path = "../../Datasets/partition_LP.json"
with open(json_file_path, 'r') as j:
     partition_LP = json.loads(j.read())

json_file_path = "../../Datasets/labels_LP.json"
with open(json_file_path, 'r') as j:
     labels_LP = json.loads(j.read())

input_shape = (224, 224, 3)
training_data_generator = data_generator.DataGenerator(partition_LP['train'], labels_LP,\
     batch_size=5, input_shape=input_shape, shuffle=False)

validation_data_generator = data_generator.DataGenerator(partition_LP['valid'], labels_LP,\
     batch_size=5, input_shape=input_shape, shuffle=False)


image, label = training_data_generator.get_one_instance('300W-LP/300W_LP/AFW/AFW_134212_1_2')

bfm_info = sio.loadmat('../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat')

pose_para = np.ravel(bfm_info['Pose_Para'])
pose_3dmm = np.ravel(pose_to_3DMM(pose_para))
shape_para = np.ravel(bfm_info['Shape_Para'][0:40])
exp_para = np.ravel(bfm_info['Exp_Para'][0:10])

pose_3dmm = np.expand_dims(pose_3dmm, 0)
shape_para = np.expand_dims(shape_para, 0)
exp_para = np.expand_dims(exp_para, 0)

pca = PCA()
pca.build()
vertices_tf = pca.call(pose_3dmm, exp_para, shape_para)
vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
vertices = tf.make_ndarray(vertices)
#print(vertices.shape)
landmarks_pred = vertices

plot_landmarks_pred(image, landmarks_pred, 'pred')
plot_landmarks_gt(image, label, name='gt')

model = create_synergy((224, 224, 3))
#print(model.summary())

loss = Synergy_Loss()
