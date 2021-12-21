import tensorflow as tf
from losses import ParameterLoss
from model.synergy import Synergy
from utils.data_utils.plotting_data import plot_landmarks, plot_pose

from model.morhaple_face_model import PCA
from utils.loading_data import loading_generators
from tensorflow.keras.optimizers import Adam, Nadam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
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
optimizer = Nadam(learning_rate=0.001)
loss_function = ParameterLoss(name='loss_Param_In', mode='normal')


losses = {
  'output_1': ParameterLoss(name='loss_Param_In', mode='normal'),
  }

model.compile(optimizer, loss_function)
print(model.summary())
model.fit(images, y, verbose=1, epochs=300)

DMM = model.predict(images)[0]

poses_pred = DMM[:12]

y_DMM = y[0]
poses_gt = y_DMM[:12]

pca = PCA(input_shape)
vertices_tf = pca(DMM)
vertices_pred = vertices_tf.numpy()

vertices_tf = pca(y_DMM)
vertices_gt = vertices_tf.numpy()

for i in range(len(list_ids)):
  plot_landmarks(images[i], vertices_pred[i], 'sanity_lmk_pred_'+str(i))
  plot_landmarks(images[i], vertices_gt[i], name='sanity_lmk_gt_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images[i], poses_pred[i], name='sanity_poses_pred_'+str(i))
  plot_pose(images[i], poses_gt[i], name='sanity_poses_gt_'+str(i))
