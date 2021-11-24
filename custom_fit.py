import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
import time

def train(model, train_dataset, valid_dataset, epochs, loss_function, optimizer):
    best_loss = np.inf
    for epoch in range(epochs):
        train_loss_list = []
        valid_loss_list = [] 
        start_time_train = time.time()
        tf.print("\nEpoch {}/{}".format(epoch+1, epochs))
        pb_1 = Progbar(len(train_dataset.list_IDS)/train_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(train_dataset):
            train_loss = train_batch(X_batch, y_batch, optimizer, loss_function, model)
            train_loss_list.append(train_loss)
        epoch_train_loss = tf.reduce_mean(train_loss_list)
        end_time_train = start_time_train - time.time()

        start_time_valid = time.time()
        pb_2 = Progbar(len(valid_dataset.list_IDS)/valid_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(valid_dataset):
            valid_loss = validation_batch(X_batch, y_batch, model, loss_function)
            valid_loss_list.append(valid_loss)
        epoch_valid_loss = tf.reduce_mean(valid_loss_list)
        end_time_valid = start_time_valid - time.time()

        if epoch_valid_loss < best_loss:
            tf.print(" ")
            model.save("model.h5")


def train_batch(X, y_true, optimizer, loss_function, model):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss_value = loss_function(y_true, y_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

def validation_batch(X, y_true, model, loss_function):
    y_pred = model(X, training=False)
    loss_value = loss_function(y_true, y_pred)
    return loss_value