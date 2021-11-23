
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.train import AdamOptimizer
tf.compat.v1.enable_eager_execution()

from losses import Synergy_Loss
from data_generator import data_generator
from model.synergy import create_synergy
from data_generator import data_generator
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

json_file_path = "../../Datasets/partition_LP.json"
with open(json_file_path, 'r') as j:
     partition_LP = json.loads(j.read())

json_file_path = "../../Datasets/labels_LP.json"
with open(json_file_path, 'r') as j:
     labels_LP = json.loads(j.read())

input_shape = (224, 224, 3)
training_data_generator = data_generator.DataGenerator(partition_LP['train'], labels_LP,\
     batch_size=5, input_shape=input_shape, shuffle=False)

validation_data_generator = data_generator.DataGenerator(partition_LP['valid'], labels_LP,\
     batch_size=5, input_shape=input_shape, shuffle=False)

model = create_synergy((224, 224, 3))
model.compile(optimizer= AdamOptimizer(learning_rate=1e-5), loss=Synergy_Loss())
print(model.summary())

model.fit(training_data_generator)

model_checkpoint_callback = ModelCheckpoint(filepath='.',
     save_weights_only=True,
     monitor='val_loss',
     mode='min',
     save_best_only=True,
     verbose=1)

model_fit = model.fit(x=training_data_generator,
validation_data=validation_data_generator,
epochs=10, 
verbose=1,
callbacks=[model_checkpoint_callback])   