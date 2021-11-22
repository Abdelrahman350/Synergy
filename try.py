from data_generator.image_utils import resize_image
from losses import Synergy_Loss
from model.synergy import create_synergy
from model.morhaple_face_model import PCA
import numpy as np
from utils.data_utils.plotting_data import *
from data_generator.preprocessing_labels import *
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

training_data_generator = data_generator.DataGenerator(partition_LP['train'], labels_LP,\
     batch_size=5, input_shape=(64, 64, 3), shuffle=False)

validation_data_generator = data_generator.DataGenerator(partition_LP['valid'], labels_LP,\
     batch_size=5, input_shape=(64, 64, 3), shuffle=False)


image, label = training_data_generator.get_one_instance('300W-LP/300W_LP/AFW/AFW_134212_1_2')

def draw_landmarks_(image_original, pt2d):
    image = image_original.copy()
    for i in range(pt2d.shape[0]):
        cv2.circle(image, (int(round(pt2d[i, 0])), int(round(pt2d[i, 1]))), 2, (0, 0, 1), -1)
    return image

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,1), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (1, 1, 1), 1)
    return image

def plot_landmarks_try(image, pts, name='foo'):
    image = plot_kpt(image, pts)
    cv2.imwrite(f"output_landmarks_try.jpg", image*255)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10, 7))
    plt.imshow(image)
    plt.xlabel("0")
    plt.ylabel("1")
    plt.scatter(pts[:, 0], pts[:, 1], color = "red", linewidths=0.1)
    plt.savefig(name+'.png')

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
    H = np.diagflat([1., 1., 1., 1.])
    H[1, 1] = -1.0
    H[1, 3] = height
    T = H.dot(T)
    return T.astype(np.float32)

bfm_info = sio.loadmat('../../Datasets/300W-LP/300W_LP/AFW/AFW_134212_1_2.mat')

[height, _, _] = image.shape
pose_para = np.ravel(bfm_info['Pose_Para'])
pose_3dmm = np.ravel(pose_to_3DMM(pose_para))
shape_para = np.ravel(bfm_info['Shape_Para'][0:40])
exp_para = np.ravel(bfm_info['Exp_Para'][0:10])

pose_3dmm = np.expand_dims(pose_3dmm, 0)
shape_para = np.expand_dims(shape_para, 0)
exp_para = np.expand_dims(exp_para, 0)

print()
print("_________________________________________________________")
print(pose_3dmm.shape, shape_para.shape, exp_para.shape)
print()
pca = PCA(height=height)
pca.build()
vertices_tf = pca.call(pose_3dmm, exp_para, shape_para)
vertices = tf.compat.v1.make_tensor_proto(vertices_tf)  # convert `tensor a` to a proto tensor
vertices = tf.make_ndarray(vertices)
print(vertices.shape)
image = resize_image(image)
landmarks_pred = vertices[0]
landmarks_gt =  label_to_pt2d(label)
landmarks_pred = resize_landmarks(landmarks_pred, (224/450.0, 224/450.0))
landmarks_gt = resize_landmarks(landmarks_gt, (224/450.0, 224/450.0))

plot_landmarks_try(image, landmarks_pred, 'pred')
plot_landmarks_try(image, landmarks_gt, 'gt')

model = create_synergy((224, 224, 3))
print(model.summary())

loss = Synergy_Loss()
