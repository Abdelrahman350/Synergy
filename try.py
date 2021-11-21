from model.synergy import create_synergy
from model.morhaple_face_model import PCA
import numpy as np
import pickle
#labels_LP['300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0'].keys()
from utils.data_utils.plotting_data import *
from data_generator.preprocessing_labels import *
from data_generator import data_generator

import json
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

json_file_path = "../../Datasets/partition_LP.json"
with open(json_file_path, 'r') as j:
     partition_LP = json.loads(j.read())

json_file_path = "../../Datasets/labels_LP.json"
with open(json_file_path, 'r') as j:
     labels_LP = json.loads(j.read())

training_data_generator = data_generator.DataGenerator(partition_LP['train'], labels_LP,\
     batch_size=5, input_shape=(64, 64, 3), shuffle=False)

validation_data_generator = data_generator.DataGenerator(partition_LP['valid'], labels_LP,\
     batch_size=5, input_shape=(64, 64, 3), shuffle=False)


image, label = training_data_generator.get_one_instance('300W-LP/300W_LP/AFW/AFW_134212_1_2')

def draw_landmarks_(image_original, pt2d):
    image = image_original.copy()
    for point in pt2d:
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 2, (0, 0, 1), -1)
    return image

def plot_landmarks_try(image, pts):
    image = draw_landmarks_(image, pts)
    cv2.imwrite(f"output_landmarks_try.jpg", image*255)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10, 7))
    plt.imshow(image)
    plt.xlabel("0")
    plt.ylabel("1")
    plt.scatter(pts[:, 0], pts[:, 1], color = "red", linewidths=0.1)
    plt.savefig('foo.png')

def get_transform_matrix(s, angles, t, height):
    """
    :param s: scale
    :param angles: [3] rad
    :param t: [3]
    :return: 4x4 transmatrix
    """
    R = eulerAngles_to_RotationMatrix(angles)
    print('inv_angles: ', rotationMatrix_to_EulerAngles(R))
    R = R.astype(np.float32)
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[3, 3] = 1.
    # scale
    S = np.diagflat([s, s, s, 1.])
    T = S.dot(T)
    # offset move
    M = np.diagflat([1., 1., 1., 1.])
    M[0:3, 3] = t.astype(np.float32)
    T = M.dot(T)
    # revert height
    # x[:,1]=height-x[:,1]
    H = np.diagflat([1., 1., 1., 1.])
    H[1, 1] = -1.0
    H[1, 3] = height
    T = H.dot(T)
    return T.astype(np.float32)

import scipy.io as sio

bfm_info = sio.loadmat('../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat')

[height, _, _] = image.shape
pose_para = np.ravel(bfm_info['Pose_Para'])
pose_3dmm = np.ravel(pose_to_3DMM(pose_para))
shape_para = np.ravel(bfm_info['Shape_Para'][0:40])
exp_para = np.ravel(bfm_info['Exp_Para'][0:10])


# pca = PCA(height=height)
# vertices_tf = pca.call(pose_3dmm, exp_para, shape_para)
# vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
# vertices = tf.make_ndarray(vertices)
# print(vertices.shape)


# plot_landmarks_try(image, vertices)

model = create_synergy((224, 224, 3))
print(model.summary())