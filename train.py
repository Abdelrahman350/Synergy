import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.train import AdamOptimizer
tf.compat.v1.enable_eager_execution()

from losses import Synergy_Loss
from utils.loading_data import loading_generators
from model.synergy import create_synergy
from data_generator import data_generator
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=(224, 224, 3), batch_size=16, shuffle=True)

model = create_synergy((224, 224, 3))
model.compile(optimizer= AdamOptimizer(learning_rate=0.02), loss=Synergy_Loss())

model_checkpoint_callback = ModelCheckpoint(filepath='.',
     save_weights_only=True,
     monitor='val_loss',
     mode='min',
     save_best_only=True,
     verbose=1)

print(model.summary())

model_fit = model.fit(x=training_data_generator,
validation_data=validation_data_generator,
epochs=10, 
verbose=1,
callbacks=[model_checkpoint_callback])   