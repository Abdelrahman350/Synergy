from os import name
from model.backbone import create_MobileNetV2
from model.morhaple_face_model import PCA
from model.encoder import MAFA
from model.decoder import Landmarks_to_3DMM
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten


def create_synergy(input_shape, num_classes=62, num_points=68):
    inputs = Input(shape=input_shape)
    X = create_MobileNetV2(input_shape=input_shape, classes=num_classes)(inputs)
    X_flattened = Flatten(name='Flatten')(X)
    Z = GlobalAveragePooling2D(name='Global_Avg_Pooling')(X)
    X_p = Dropout(0.2)(X_flattened)
    pose_3DMM = Dense(name='pose_3DMM', units=12)(X_p)
    X_exp = Dropout(0.2)(X_flattened)
    alpha_exp = Dense(name='alpha_exp', units=10)(X_exp)
    X_shape = Dropout(0.2)(X_flattened)
    alpha_shp = Dense(name='alpha_shp', units=40)(X_shape)

    morphable_model = PCA(input_shape=input_shape, name='Morphable_layer')
    
    Lc = morphable_model(pose_3DMM, alpha_exp, alpha_shp)
    Lr = MAFA(num_points=num_points)(Lc, Z, alpha_exp, alpha_shp)
    pose_3DMM_hat, alpha_exp, alpha_shp = Landmarks_to_3DMM(num_classes=num_classes,\
         num_points=num_points)(Lr)

    model = Model(inputs=[inputs],\
          outputs=[pose_3DMM, Lc, pose_3DMM_hat], name='Synergy')
    return model

class Synergy(Model):
      def __init__(self, input_shape, num_classes=62, num_points=68, **kwargs):
            super(Synergy, self).__init__(**kwargs, name="Synergy")
            self.input_shape_ = input_shape
            self.mobileNet = create_MobileNetV2(input_shape=input_shape, classes=num_classes)
            self.flatten = Flatten(name='Flatten')
            self.GlobalAvgBooling = GlobalAveragePooling2D(name='Global_Avg_Pooling')
          
            self.dropOut_pose = Dropout(0.2, name='pose_dropout')
            self.dense_pose = Dense(name='pose_3DMM', units=12)

            self.dropOut_exp = Dropout(0.2, name='exp_dropout')
            self.dense_exp = Dense(name='alpha_exp', units=10)

            self.dropOut_shp = Dropout(0.2, name='shp_dropout')
            self.dense_shp = Dense(name='alpha_shp', units=40)
            self.morphable_model = PCA(input_shape=input_shape, name='Morphable_layer')

            self.encoder =  MAFA(num_points=num_points)
            self.decoder = Landmarks_to_3DMM(num_classes=num_classes, num_points=num_points)

      def call(self, batch_images):
            X = self.mobileNet(batch_images)
            X_flattened = self.flatten(X)
            Z = self.GlobalAvgBooling(X)

            X_pose = self.dropOut_pose(X_flattened)
            pose_3DMM = self.dense_pose(X_pose)

            X_exp = self.dropOut_exp(X_flattened)
            alpha_exp = self.dense_exp(X_exp)

            X_shp = self.dropOut_shp(X_flattened)
            alpha_shp = self.dense_shp(X_shp)

            Lc = self.morphable_model(pose_3DMM, alpha_exp, alpha_shp)
            Lr = self.encoder(Lc, Z, alpha_exp, alpha_shp)
            pose_3DMM_hat, alpha_exp, alpha_shp = self.decoder(Lr)

            return pose_3DMM, Lc, pose_3DMM_hat
      
      def model(self):
        images = Input(shape=self.input_shape_, name='Input_Images')
        return Model(inputs=[images], outputs=self.call(images))