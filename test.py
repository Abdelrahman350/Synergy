import tensorflow as tf
from utils.data_utils.plotting_data import plot_landmarks, plot_pose
from model.synergy import Synergy

from model.morhaple_face_model import PCA
from losses import ParameterLoss, WingLoss
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam

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

input_shape = (450, 450, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2", "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0",\
  "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
  "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

model = Synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.0001)

losses = {
  'Param': ParameterLoss(name='loss_Param_In', mode='normal'),
  'Param*': ParameterLoss(name='loss_Param_S2', mode='3dmm'),
  'Lc': WingLoss(name='loss_LMK_f0'),
  'Lr': WingLoss(name='loss_LMK_pointNet')
  }

loss_weights = {'Param':0.02, 'Param*':0.02, 'Lc':0.05, 'Lr':0.05}
model.compile(optimizer, losses, loss_weights=loss_weights)
print(model.summary())
model.load_weights("checkpoints/model")

DMM = model.predict(images)
poses_pred = DMM['Param']

y_DMM = y['Param']
poses_gt = y_DMM

pca = PCA(input_shape)
vertices_tf = pca(poses_pred)
vertices_pred = vertices_tf.numpy()

vertices_tf = pca(y_DMM)
vertices_gt = vertices_tf.numpy()

for i in range(len(list_ids)):
  plot_landmarks(images[i], vertices_pred[i], 'test_lmk_pred_'+str(i))
  plot_landmarks(images[i], vertices_gt[i], name='test_lmk_gt_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images[i], poses_pred[i], name='test_poses_pred_'+str(i))
  plot_pose(images[i], poses_gt[i], name='test_poses_gt_'+str(i))
