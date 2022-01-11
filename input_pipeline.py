
from model.synergy import Synergy
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from utils.loading_data import loading_generators

set_GPU()
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)

test = '300w'
if test == 'AFLW':
  training_data_generator, validation_data_generator = loading_generators(dataset='AFLW',\
        input_shape=input_shape, batch_size=32, shuffle=True)
  list_ids = ["AFLW2000-3D/AFLW2000/image00002", "AFLW2000-3D/AFLW2000/image00004",\
    "AFLW2000-3D/AFLW2000/image00006", "AFLW2000-3D/AFLW2000/image00008",
    "AFLW2000-3D/AFLW2000/image00010"]
elif test == '300w':
  training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
        input_shape=input_shape, batch_size=32, shuffle=True)
  list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
      "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
        "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
training_data_generator.augmentation = False
images, y = training_data_generator.data_generation(list_ids)


images_ori = []
dataset_path='../../Datasets/300W_AFLW/'
for i in range(len(list_ids)):
  import numpy as np
  image = np.zeros((450, 450, 3))
  image[0:images[i].shape[0], 0:images[i].shape[1], :] = images[i]
  images_ori.append(image)

y_DMM = y['Pm']
poses_gt = y_DMM
vertices = y['Lc']

for i in range(len(list_ids)):
  plot_landmarks(images_ori[i], vertices[i], 'lmk_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images_ori[i], poses_gt[i], vertices[i], 'pose_'+str(i))

model = Synergy(input_shape=input_shape)
model.summary()