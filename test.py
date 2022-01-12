import tensorflow as tf
from set_tensorflow_configs import set_GPU
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy

from model.morhaple_face_model import PCA
from losses import ParameterLoss, WingLoss
from utils.loading_data import loading_generators
import cv2

set_GPU()
IMG_H = 128
input_shape = (IMG_H, IMG_H, 3)
model_path = "checkpoints/model"
test = "AFLW"

if test == 'AFLW':
  training_data_generator, validation_data_generator = loading_generators(dataset='AFLW',\
        input_shape=input_shape, batch_size=32, shuffle=True)
  list_ids = ["AFLW2000-3D/AFLW2000/image01986", "AFLW2000-3D/AFLW2000/image00405",\
     "AFLW2000-3D/AFLW2000/image00291", "AFLW2000-3D/AFLW2000/image02522",\
        "AFLW2000-3D/AFLW2000/image04269", "AFLW2000-3D/AFLW2000/image03515",\
           "AFLW2000-3D/AFLW2000/image02183", "AFLW2000-3D/AFLW2000/image04102",\
              "AFLW2000-3D/AFLW2000/image01079", "AFLW2000-3D/AFLW2000/image00187",\
                 "AFLW2000-3D/AFLW2000/image00359", "AFLW2000-3D/AFLW2000/image04188",\
                    "AFLW2000-3D/AFLW2000/image02243", "AFLW2000-3D/AFLW2000/image00053",\
                       "AFLW2000-3D/AFLW2000/image02213", "AFLW2000-3D/AFLW2000/image04004",
                        "AFLW2000-3D/AFLW2000/image03043", "AFLW2000-3D/AFLW2000/image01366",
                         "AFLW2000-3D/AFLW2000/image02782", "AFLW2000-3D/AFLW2000/image02664",
                          "AFLW2000-3D/AFLW2000/image00741", "AFLW2000-3D/AFLW2000/image00771",
                           "AFLW2000-3D/AFLW2000/image00062", "AFLW2000-3D/AFLW2000/image00554", "AFLW2000-3D/AFLW2000/image03077", "AFLW2000-3D/AFLW2000/image03705", "AFLW2000-3D/AFLW2000/image02597", "AFLW2000-3D/AFLW2000/image01981", "AFLW2000-3D/AFLW2000/image03273", "AFLW2000-3D/AFLW2000/image02918", "AFLW2000-3D/AFLW2000/image03640", "AFLW2000-3D/AFLW2000/image01427", "AFLW2000-3D/AFLW2000/image01449", "AFLW2000-3D/AFLW2000/image00922", "AFLW2000-3D/AFLW2000/image03375", "AFLW2000-3D/AFLW2000/image01688", "AFLW2000-3D/AFLW2000/image02038", "AFLW2000-3D/AFLW2000/image03479", "AFLW2000-3D/AFLW2000/image01110", "AFLW2000-3D/AFLW2000/image03897", "AFLW2000-3D/AFLW2000/image01649", "AFLW2000-3D/AFLW2000/image03561", "AFLW2000-3D/AFLW2000/image00809", "AFLW2000-3D/AFLW2000/image00060",]
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
for id in list_ids:
  image_path = dataset_path + id + '.jpg'
  image = cv2.imread(image_path)
  image = image.astype(float)
  image /= 255.0
  images_ori.append(image)

model = Synergy(input_shape=input_shape)

print(model.summary())
model.load_weights("checkpoints/model")

DMM = model.predict(images)
poses_pred = DMM['Pm']

y_DMM = y['Pm']
poses_gt = y_DMM

pca = PCA(input_shape)
vertices_tf = pca(poses_pred)
vertices_pred = vertices_tf.numpy()

vertices_gt = y['Lc']

for i in range(len(list_ids)):
  plot_landmarks(images[i], vertices_gt[i], name='test_lmk_gt_'+str(i))
  plot_landmarks(images[i], vertices_pred[i], 'test_lmk_pred_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images_ori[i], poses_gt[i], vertices_gt[i], name='test_poses_gt_'+str(i))
  plot_pose(images_ori[i], poses_pred[i], vertices_pred[i], name='test_poses_pred_'+str(i))