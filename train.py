
#labels_LP['300W-LP/300W_LP/HELEN_Flip/HELEN_1269874180_1_0'].keys()
from utils.data_utils.plotting_data import *
from data_generator.preprocessing_labels import label_3DDm_to_pose, label_3DDm_to_pt2d
from data_generator import data_generator
from model.encoder import MMFA
from model.decoder import Landmarks_to_3DMM
import json
import tensorflow as tf

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

# model1 = MMFA()
# print(model1.summary())
# tf.keras.utils.plot_model(model1, "encoder.png")

# model2 = Landmarks_to_3DMM()
# print(model2.summary())
# tf.keras.utils.plot_model(model2, "decoder.png")
#print(*label_3DDm_to_pose(label))
plot_pose_image(image, label)
#print(label_3DDm_to_pt2d(label))
plot_landmarks_image(image, label)
