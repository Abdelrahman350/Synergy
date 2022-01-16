import tensorflow as tf
from losses import ParameterLoss
from model.backbone import create_MobileNetV2
from model.morhaple_face_model import PCA
from model.encoder import MAFA
from model.decoder import Landmarks_to_3DMM
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout

class Synergy(Model):
      def __init__(self, input_shape, num_classes=62, num_points=68, **kwargs):
            super(Synergy, self).__init__(**kwargs, name="Synergy")
            self.input_shape_ = input_shape
            self.mobileNet = create_MobileNetV2(input_shape=input_shape, classes=num_classes)
            self.GlobalAvgBooling = GlobalAveragePooling2D(name='Global_Avg_Pooling')
          
            self.dropOut_pose = Dropout(0.2, name='pose_dropout')
            self.dense_pose = Dense(name='pose_3DMM', units=12)

            self.dropOut_shp = Dropout(0.2, name='shp_dropout')
            self.dense_shp = Dense(name='alpha_shp', units=40)

            self.dropOut_exp = Dropout(0.2, name='exp_dropout')
            self.dense_exp = Dense(name='alpha_exp', units=10)

            self.morphable_model = PCA(input_shape=input_shape, name='Morphable_layer')

            self.encoder = MAFA(num_points=num_points)
            self.decoder = Landmarks_to_3DMM(num_points=num_points)
            self.paramLoss = ParameterLoss(name='loss_Param_S1S2', mode='3dmm')

      def call(self, batch_images):
            X = self.mobileNet(batch_images)
            X = self.GlobalAvgBooling(X)
            Z = tf.identity(X, name='Global_Average_Pooling')

            X_pose = self.dropOut_pose(X)
            pose_3DMM = self.dense_pose(X_pose)

            X_shp = self.dropOut_shp(X)
            alpha_shp = self.dense_shp(X_shp)

            X_exp = self.dropOut_exp(X)
            alpha_exp = self.dense_exp(X_exp)

            Param_3D = tf.concat((pose_3DMM, alpha_shp, alpha_exp), axis=-1)
            Lc = self.morphable_model(Param_3D)
            point_residual = self.encoder(Lc, Z, Param_3D[:, 12:52], Param_3D[:, 52:])
            Lr = tf.add(0.05*point_residual, Lc, name='point_residual')
            Param_3D_hat = self.decoder(Lr)
            Lg = self.paramLoss(Param_3D, Param_3D_hat)
            self.add_loss(0.001 * Lg)
            return {'Pm':Param_3D, 'Pm*':Param_3D_hat, 'Lc':Lc, 'Lr':Lr}
      
      def model(self):
            images = Input(shape=self.input_shape_, name='Input_Images')
            return Model(inputs=[images], outputs=self.call(images), name='Synergy')
      
      def summary(self):
            return self.model().summary()

      def get_config(self):
            base_config = super(Synergy, self).get_config()
            return {**base_config, 
                        'input_shape': self.input_shape_,
                        'backbone': self.mobileNet,
                        'flatten': self.flatten,
                        'GlobalAvgBooling': self.GlobalAvgBooling,
                        'dropOut_pose': self.dropOut_pose,
                        'dense_pose': self.dense_pose,
                        'dropOut_shp': self.dropOut_shp,
                        'dense_shp': self.dense_shp,
                        'dropOut_exp': self.dropOut_exp,
                        'dense_exp': self.dense_exp,
                        'morphable_model': self.morphable_model,
                        'encoder': self.encoder,
                        'decoder': self.decoder,
                        'paramLoss': self.paramLoss
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
