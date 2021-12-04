import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
import wandb

def train(model, train_dataset, valid_dataset, epochs, loss_function, optimizer, load_model=True):
    best_loss = np.inf
    if load_model:
        model.build((1, train_dataset.input_shape[0],\
            train_dataset.input_shape[1], train_dataset.input_shape[2]))
        model.load_weights("checkpoints/Model.h5")
        cumulative_valid_loss = 0
        for batch, (X_batch, y_batch) in enumerate(valid_dataset):
            valid_loss = validation_batch(X_batch,\
                 y_batch, model, loss_function)/valid_dataset.batch_size
            cumulative_valid_loss += valid_loss
            last_batch = batch
        cumulative_valid_loss /= (last_batch+1)
        best_loss = cumulative_valid_loss
        tf.print(f"Loaded model with validation_loss = {best_loss}\n")
    
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        cumulative_train_loss = 0
        cumulative_valid_loss = 0
        last_batch = 0
        tf.print("\nEpoch {}/{}:\n".format(epoch+1, epochs))
        pb_1 = Progbar(len(train_dataset.list_IDs)/train_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(train_dataset):
            train_loss = train_batch(X_batch,\
                 y_batch, optimizer, loss_function, model)/train_dataset.batch_size
            values=[('train_loss', train_loss)]
            cumulative_train_loss += train_loss
            pb_1.update(batch, values)
            last_batch = batch
        cumulative_train_loss /= (last_batch+1)
        values=[('train_loss', cumulative_train_loss)]
        pb_1.update(len(train_dataset.list_IDs)/train_dataset.batch_size, values=values)

        pb_2 = Progbar(len(valid_dataset.list_IDs)/valid_dataset.batch_size,\
            stateful_metrics=None)
        for batch, (X_batch, y_batch) in enumerate(valid_dataset):
            valid_loss = validation_batch(X_batch,\
                 y_batch, model, loss_function)/valid_dataset.batch_size
            values=[('valid_loss', valid_loss)]
            pb_2.update(batch, values)
            cumulative_valid_loss += valid_loss
            last_batch = batch
        cumulative_valid_loss /= (last_batch+1)
        values=[('train_loss', cumulative_train_loss), ('val_loss', cumulative_valid_loss)]
        pb_2.update(len(valid_dataset.list_IDs)/valid_dataset.batch_size, values=values)

        wandb.log({'epoch': epoch, 'training loss': cumulative_train_loss,\
             'validation loss': cumulative_valid_loss})         
        if cumulative_valid_loss < best_loss:
            model_name = "Model.h5"
            model.save_weights("checkpoints/" + model_name)
            print(f"The validation loss improved from {best_loss} to {cumulative_valid_loss}.\
                 Saving model {model_name}")
            best_loss = cumulative_valid_loss

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