
from model.synergy import Synergy
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from utils.loading_data import loading_test_examples
import cv2

set_GPU()
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
test = '300W'

list_ids, training_data_generator, validation_data_generator = loading_test_examples(test, input_shape)
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)

images_ori = []

for i in range(len(list_ids)):
  import numpy as np
  image = np.zeros((450, 450, 3))
  image[0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]
  images_ori.append(image)

y_DMM = y['Pm']
poses_gt = y_DMM
vertices = y['Lc']

for i in range(len(list_ids)):
  image = plot_landmarks(images_ori[i], vertices[i])
  cv2.imwrite('output/lmk_'+str(i)+'.jpg', image)

for i in range(len(list_ids)):
  image = plot_pose(images_ori[i], poses_gt[i], vertices[i])
  cv2.imwrite('output/pose_'+str(i)+'.jpg', image)

model = Synergy(input_shape=input_shape)
model.summary()