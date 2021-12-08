import tensorflow as tf
from model.synergy import Synergy, create_synergy
from utils.data_utils.plotting_data import plot_landmarks, plot_pose

from model.morhaple_face_model import PCA
from utils.custom_fit import train, train_on_image
from losses import Synergy_Loss
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

input_shape = (224, 224, 3)
training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=input_shape, batch_size=32, shuffle=True)

list_ids = ["300W-LP/300W_LP/AFW/AFW_134212_1_2"]#, "300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0"]#,\
    #  "300W-LP/300W_LP/AFW/AFW_4512714865_1_3", "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0737_13",
    #   "300W-LP/300W_LP/LFPW_Flip/LFPW_image_train_0047_4"]
images, y = training_data_generator.data_generation(list_ids)

model = create_synergy(input_shape=input_shape)
optimizer = Nadam(learning_rate=0.1)
loss_function = Synergy_Loss()
# train_on_image(model, images, y, 5000, loss_function, optimizer, False)

model.compile(optimizer, loss_function)
print(model.summary())
model.fit(images, y, verbose=1, epochs=10000)

DMM = model.predict(images)

poses_pred = DMM[:, 0:12]
poses_gt = y[:, 0:12]

print("GT = ", poses_gt)
print()
print("Pred = ", poses_pred)
# poses_hat = poses_tf.numpy()
# vertices = vertices_tf.numpy()

# for i in range(len(list_ids)):
#     plot_landmarks(images[i], vertices[i], 'sanity_lmk_pred_'+str(i))
#     print(vertices[i])
#     plot_landmarks(images[i], y[1][i], name='sanity_lmk_gt_'+str(i))

for i in range(len(list_ids)):
  plot_pose(images[i], poses_pred[i], name='sanity_poses_pred_'+str(i))
  plot_pose(images[i], poses_gt[i], name='sanity_poses_gt_'+str(i))

# from matplotlib import pyplot as plt
# plt.scatter(vertices[:, 0], vertices[:, 1])
# plt.savefig('foo.png')