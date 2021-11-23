from os import name
from model.backbone import create_MobileNetV2
from model.morhaple_face_model import PCA
from model.encoder import MAFA
from model.decoder import Landmarks_to_3DMM
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten

# Landmarks_to_3DMM()
# PCA()
# create_MobileNetV2()

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
    morphable_model.build()
    Lc = morphable_model.call(pose_3DMM, alpha_exp, alpha_shp)
    Z = tf.expand_dims(Z, 1)
    alpha_exp = tf.expand_dims(alpha_exp, 1)
    alpha_shp = tf.expand_dims(alpha_shp, 1)
    Lr = MAFA(num_points=num_points)(Lc, Z, alpha_exp, alpha_shp)
    pose_3DMM_hat, alpha_exp, alpha_shp = Landmarks_to_3DMM(num_classes=num_classes,\
         num_points=num_points)(Lr)
    
    model = Model(inputs=[inputs],\
          outputs=[pose_3DMM, Lr, pose_3DMM_hat, alpha_exp, alpha_shp], name='Synergy')
    return model
