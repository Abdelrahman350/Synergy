import os.path as osp
from model.BFM import PCA
import numpy as np
import pickle
#labels_LP['300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0'].keys()
from utils.data_utils.plotting_data import *
from data_generator.preprocessing_labels import eulerAngles_to_RotationMatrix, label_3DDm_to_pose, label_3DDm_to_pt2d, label_to_pt2d, pose_3DMM_to_fPt, rotationMatrix_to_EulerAngles
from data_generator import data_generator
from model.encoder import MMFA
from model.decoder import Landmarks_to_3DMM
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

# print(labels_LP['300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0'])
#300W-LP/300W_LP/AFW/AFW_134212_1_2 #300W-LP/300W_LP/AFW/AFW_3989161_1_0
image, label = training_data_generator.get_one_instance('300W-LP/300W_LP/AFW/AFW_134212_1_2')

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

d = make_abs_path('3dmm_data')
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))

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

# PCA basis for shape, expression, texture
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))
w_tex = _load(osp.join(d, 'w_tex_sim.npy'))[:,:40]
# param_mean and param_std are used for re-whitening
meta = _load(osp.join(d, 'param_whitening.pkl'))
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
# mean values
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u_tex = _load(osp.join(d, 'u_tex.npy'))
# print(f"param_mean.shape:{param_mean.shape},  param_std.shape:{param_std.shape}")

# print("keypoints.shape: ", keypoints.shape)
# print(f"w_shp.shape:{w_shp.shape},  w_exp.shape:{w_exp.shape}")
# print(f"u_shp.shape:{u_shp.shape},  u_exp.shape:{u_exp.shape}")

u = u_shp + u_exp
u_base = u[keypoints]
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]

# print("u.shape: ", u.shape)
# print("u_base.shape: ", u_base.shape)
# print("w_shp_base.shape: ", w_shp_base.shape)
# print("w_exp_base.shape: ", w_exp_base.shape)

# print('-----------------------------------------------')
#label = label * param_std + param_mean

# print(t.shape)
# print("alpha_exp.shape: ", alpha_exp.shape)
# print("alpha_Shape.shape: ", alpha_Shape.shape)

# vertex = f*R @ (u_base +\
#      w_shp_base @ alpha_Shape.T + w_exp_base @ alpha_exp.T).reshape(3, -1, order='F') + t

# print('-----------------------------------------------')
# print(vertex.shape)
# #test = R@(w_shp_base @ alpha_Shape.T).reshape(3, -1, order='F')
# print(vertex.T)
# print()
# print(label_to_pt2d(label))

# vertex = u_base +\
#      w_shp_base @ alpha_Shape.T + w_exp_base @ alpha_exp.T
import scipy.io as sio

bfm_info = sio.loadmat('../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat')

[height, _, _] = image.shape
pose_para = bfm_info['Pose_Para'].T.astype(np.float32)
shape_para = bfm_info['Shape_Para'][0:40].astype(np.float32)
exp_para = bfm_info['Exp_Para'][0:10].astype(np.float32)

#vertices = u_base + w_shp_base.dot(shape_para) + w_exp_base.dot(exp_para)
pca = PCA(height=height)
pca.build()
vertices_tf = pca.call(pose_para, exp_para, shape_para)
vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
vertices = tf.make_ndarray(vertices)
print(vertices.shape)
# vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T
# print('_____________________________________________')
# print(vertices[0:5])
# print()
# print(tf.transpose(tf.reshape(vertices_tf, (int(len(vertices_tf)/3), 3)))[0:5])
# print('_____________________________________________')
s = pose_para[-1, 0]
angles = pose_para[:3, 0]
print('True angles: ', angles)
print('true_inv: ', label_3DDm_to_pose(label))
t = pose_para[3:6, 0]

# T_bfm = get_transform_matrix(s, angles, t, height)
# temp_ones_vec = np.ones((len(vertices), 1))
# homo_vertices = np.concatenate((vertices, temp_ones_vec), axis=-1)
# image_vertices = homo_vertices.dot(T_bfm.T)[:, 0:3]

plot_landmarks_try(image, vertices)

