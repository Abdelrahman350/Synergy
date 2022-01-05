import tensorflow as tf
from data_generator.data_generator import DataGenerator
from model.synergy import Synergy
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose

from model.morhaple_face_model import PCA
from utils.loading_data import loading_generators
import cv2

set_GPU()

test = '300w'
IMG_H = 450

input_shape = (IMG_H, IMG_H, 3)

train_dataset = DataGenerator(
        root="../../Datasets/",
        filelists='../../Datasets/3dmm_data/',
        param_fp='3dmm_data/param_all_norm_v201.pkl',
        gt_transform=False,
        transform=None
    )

images_ori = []
dataset_path='../../Datasets/300W_AFLW_Augmented/'
list_ids = [0, 1, 2]
for id in list_ids:
  image_path = dataset_path + id + '.jpg'
  image = cv2.imread(image_path)
  image = image.astype(float)
  image /= 127.5
  image -= 1
  images_ori.append(image)

y_DMM = y['Pm']
poses_gt = y_DMM

pca = PCA(input_shape)
vertices_tf = pca(y_DMM)
vertices = vertices_tf.numpy()*450.0/input_shape[0]

for i in range(len(list_ids)):
  plot_landmarks(images_ori[i], vertices[i], 'lmk_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images_ori[i], poses_gt[i], 'pose_'+str(i))

model = Synergy(input_shape=input_shape)
model.summary()

# model.save_weights("checkpoints/model_synergy")
# print()

# model_test = Synergy(input_shape=input_shape)
# model_test.load_weights("checkpoints/model_synergy")
# print(model_test.summary())