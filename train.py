import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.train import AdamOptimizer

from utils.custom_fit import train
from losses import Synergy_Loss
from utils.loading_data import loading_generators
from model.synergy import create_synergy, Synergy

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

training_data_generator, validation_data_generator = loading_generators(dataset='300w',\
      input_shape=(224, 224, 3), batch_size=32, shuffle=True)
model = Synergy((224, 224, 3))
optimizer = AdamOptimizer(learning_rate=0.02)
loss_function = Synergy_Loss()
# model.compile(optimizer= AdamOptimizer(learning_rate=0.02), loss=Synergy_Loss())

# model_checkpoint_callback = ModelCheckpoint(filepath='.',
#      save_weights_only=True,
#      monitor='val_loss',
#      mode='min',
#      save_best_only=True,
#      verbose=1)

print(model.model().summary())
# model_fit = model.fit(x=training_data_generator,
# validation_data=validation_data_generator,
# epochs=10, 
# verbose=1,
# callbacks=[model_checkpoint_callback])
train(model, training_data_generator, validation_data_generator, 100,\
       loss_function, optimizer, False)