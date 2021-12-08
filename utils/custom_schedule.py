import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class MyLRSchedule(LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step + 1)
