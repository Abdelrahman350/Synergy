from os import name
from model.backbone import create_MobileNetV2
from model.morhaple_face_model import PCA
from model.encoder import MAFA
from model.decoder import Landmarks_to_3DMM, Landmarks_to_3DMM_2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten


def create_synergy(input_shape, num_classes=62, num_points=68):
    inputs = Input(shape=input_shape)
    X = create_MobileNetV2(input_shape=input_shape, classes=num_classes)(inputs)
    X = GlobalAveragePooling2D(name='Global_Avg_Pooling')(X)
    Z = tf.identity(X)
#     X_p = Dropout(0.2)(X)
    pose_3DMM = Dense(name='pose_3DMM', units=12)(X)
#     X_exp = Dropout(0.2)(X)
    alpha_exp = Dense(name='alpha_exp', units=10)(X)
#     X_shape = Dropout(0.2)(X)
    alpha_shp = Dense(name='alpha_shp', units=40)(X)
    morphable_model = PCA(input_shape=input_shape, name='Morphable_layer')
    
    Lc = morphable_model(pose_3DMM, alpha_exp, alpha_shp)
    Lr = MAFA(num_points=num_points)(Lc, Z, alpha_exp, alpha_shp)
    pose_3DMM_hat, alpha_exp, alpha_shp = Landmarks_to_3DMM(num_points=num_points)(Lr)
    model = Model(inputs=[inputs],\
          outputs=[pose_3DMM_hat, alpha_exp, alpha_shp, Lr], name='Synergy')
    return model

class Synergy(Model):
      def __init__(self, input_shape, num_classes=62, num_points=68, **kwargs):
            super(Synergy, self).__init__(**kwargs, name="Synergy")
            self.input_shape_ = input_shape
            self.mobileNet = create_MobileNetV2(input_shape=input_shape, classes=num_classes)
            self.GlobalAvgBooling = GlobalAveragePooling2D(name='Global_Avg_Pooling')
          
            self.dropOut_pose = Dropout(0.2, name='pose_dropout')
            self.dense_pose = Dense(name='pose_3DMM', units=12)

            self.dropOut_exp = Dropout(0.2, name='exp_dropout')
            self.dense_exp = Dense(name='alpha_exp', units=10)

            self.dropOut_shp = Dropout(0.2, name='shp_dropout')
            self.dense_shp = Dense(name='alpha_shp', units=40)
            self.morphable_model = PCA(input_shape=input_shape, name='Morphable_layer')

            self.encoder =  MAFA(num_points=num_points)
            self.decoder = Landmarks_to_3DMM(num_points=num_points)

      def call(self, batch_images):
            X = self.mobileNet(batch_images)
            X = self.GlobalAvgBooling(X)
            Z = tf.identity(X)

            X_pose = self.dropOut_pose(X)
            pose_3DMM = self.dense_pose(X_pose)

            X_exp = self.dropOut_exp(X)
            alpha_exp = self.dense_exp(X_exp)

            X_shp = self.dropOut_shp(X)
            alpha_shp = self.dense_shp(X_shp)

            Lc = self.morphable_model(pose_3DMM, alpha_exp, alpha_shp)
            Lr = self.encoder(Lc, Z, alpha_exp, alpha_shp)
            pose_3DMM_hat, alpha_exp, alpha_shp = self.decoder(Lr)

            return {'pose_3DMM_hat': pose_3DMM_hat, 'alpha_exp_hat': alpha_exp,\
                   'alpha_shp_hat': alpha_shp, 'Lr': Lr}
      
      def model(self):
            images = Input(shape=self.input_shape_, name='Input_Images')
            return Model(inputs=[images], outputs=self.call(images), name="Synergy")

      def get_config(self):
            base_config = super(Synergy, self).get_config()
            return {**base_config, 
                        "input_shape": self.input_shape_,
                        "backbone": self.mobileNet,
                        "flatten": self.flatten,
                        "GlobalAvgBooling": self.GlobalAvgBooling,
                        "dropOut_pose": self.dropOut_pose,
                        "dense_pose": self.dense_pose,
                        "dropOut_exp": self.dropOut_exp,
                        "dense_exp": self.dense_exp,
                        "dropOut_shp": self.dropOut_shp,
                        "dense_shp": self.dense_shp,
                        "morphable_model": self.morphable_model,
                        "encoder": self.encoder,
                        "decoder": self.decoder
                  }
      
      def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            X, y = data

            with tf.GradientTape() as tape:
                  y_pred = self(X, training=True)  # Forward pass
                  # Compute the loss value
                  # (the loss function is configured in `compile()`)
                  loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}
      
      def test_step(self, data):
            # Unpack the data
            X, y = data
            # Compute predictions
            y_pred = self(X, training=False)
            # Updates the metrics tracking the loss
            self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # Update the metrics.
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}
