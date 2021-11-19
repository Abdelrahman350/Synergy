from os import name
from model.backbone import create_MobileNetV2
from model.morhaple_face_model import PCA
from model.encoder import MMFA
from model.decoder import Landmarks_to_3DMM
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout 

# Landmarks_to_3DMM()
# PCA()
# create_MobileNetV2()

def create_synergy(input_shape, classes=62):
    inputs = Input(shape=input_shape)
    X = create_MobileNetV2(input_shape=input_shape, classes=classes)(inputs)
    Z = GlobalAveragePooling2D(name='Global_Avg_Polling')(X)
    X_p = Dropout(0.2)(X)
    pose = Dense(name='pose', units=12)(X_p)
    X_exp = Dropout(0.2)(X)
    expression = Dense(name='expression', units=10)(X_exp)
    X_shape = Dropout(0.2)(X)
    shape = Dense(name='shape', units=40)(X_shape)
    model = Model(inputs=[inputs], outputs=[Z, pose, expression, shape], name='Synergy')
    return model
