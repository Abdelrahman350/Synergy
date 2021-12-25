import tensorflow as tf
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy

from model.morhaple_face_model import PCA
from losses import ParameterLoss, WingLoss
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam
import cv2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

input_shape = (128, 128, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
  "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
  "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

images_ori = []
dataset_path='../../Datasets/'
for id in list_ids:
  image_path = dataset_path + id + '.jpg'
  image = cv2.imread(image_path)
  image = image / 255.0
  images_ori.append(image)

model = Synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.0001)

losses = {
  'Pm': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Pm*': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Pm':0.2, 'Pm*':0.02, 'Lc':0.05, 'Lr':0.05}
model.compile(optimizer, losses, loss_weights=loss_weights)
print(model.summary())
model.load_weights("checkpoints/model")

DMM = model.predict(images)
poses_pred = DMM['Pm']

y_DMM = y['Pm']
poses_gt = y_DMM

pca = PCA(input_shape)
vertices_tf = pca(poses_pred)
vertices_pred = vertices_tf.numpy()*450/128.0

vertices_tf = pca(y_DMM)
vertices_gt = vertices_tf.numpy()*450/128.0

for i in range(len(list_ids)):
  plot_landmarks(images_ori[i], vertices_pred[i], 'test_lmk_pred_'+str(i))
  plot_landmarks(images_ori[i], vertices_gt[i], name='test_lmk_gt_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images_ori[i], poses_pred[i], name='test_poses_pred_'+str(i))
  plot_pose(images_ori[i], poses_gt[i], name='test_poses_gt_'+str(i))
