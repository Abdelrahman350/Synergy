import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np

def train(model, train_dataset, valid_dataset, epochs, loss_function, optimizer):
    best_loss = np.inf
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        tf.print("\nEpoch {}/{}:\n".format(epoch+1, epochs))
        pb_1 = Progbar(len(train_dataset.list_IDs)/train_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(train_dataset):
            train_loss = train_batch(X_batch,\
                 y_batch, optimizer, loss_function, model)/train_dataset.batch_size
            values=[('train_loss', train_loss)]
            pb_1.update(batch, values)
        pb_1.update(len(train_dataset.list_IDs)/train_dataset.batch_size, values=values)

        pb_2 = Progbar(len(valid_dataset.list_IDs)/valid_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(valid_dataset):
            valid_loss = validation_batch(X_batch,\
                 y_batch, model, loss_function)/valid_dataset.batch_size
            values=[('valid_loss', valid_loss)]
            pb_2.update(batch, values)
        values=[('train_loss', train_loss), ('val_loss', valid_loss)]
        pb_2.update(len(valid_dataset.list_IDs)/valid_dataset.batch_size, values=values)
        model.save("my_h5_model.h5")

def train_batch(X, y_true, optimizer, loss_function, model):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss_value = loss_function(y_true, y_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

def validation_batch(X, y_true, model, loss_function):
    y_pred = model(X, training=True)
    loss_value = loss_function(y_true, y_pred)
    return loss_value