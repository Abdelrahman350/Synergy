import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import wandb
import numpy as np

from utils.custom_fit import train
from losses import Synergy_Loss
from utils.loading_data import loading_generators
from model.synergy import Synergy

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
      input_shape=input_shape, batch_size=64, shuffle=True)
model = Synergy(input_shape)

var = tf.Variable(np.random.random(size=(1,)))
learning_rate = ExponentialDecay(initial_learning_rate=0.03, decay_steps=20, decay_rate=0.5)
optimizer = Nadam(learning_rate=0.0007)
loss_function = Synergy_Loss()

print(model.model().summary())
experiment_name = "Synergy_mobilenetV2"
resume = True
run = wandb.init(project="Synergy", name= experiment_name, resume= resume)
wandb.save("train.py")

train(model, training_data_generator, validation_data_generator, 500,\
       loss_function, optimizer, True)