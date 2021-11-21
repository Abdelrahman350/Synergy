import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization 
from tensorflow.keras.layers import ReLU, MaxPool1D, concatenate
from tensorflow.keras import Model

def Landmarks_to_3DMM(num_classes=62, num_points=68):
    Lr = Input(shape=(num_points, 3), name='Refined_Landmarks')

    # First hidden layer
    conv1 = Conv1D(filters=64, kernel_size=1, name='Decoder_Conv1D_1')(Lr)
    bn1 = BatchNormalization(name='Decoder_BatchNormalization_1')(conv1)
    relu1 = ReLU(name='Decoder_ReLU_1')(bn1)

    # Second hidden layer
    conv2 = Conv1D(filters=64, kernel_size=1, name='Decoder_Conv1D_2')(relu1)
    bn2 = BatchNormalization(name='Decoder_BatchNormalization_2')(conv2)
    relu2 = ReLU(name='Decoder_ReLU_2')(bn2)

    # Third hidden layer
    conv3 = Conv1D(filters=128, kernel_size=1, name='Decoder_Conv1D_3')(relu2)
    bn3 = BatchNormalization(name='Decoder_BatchNormalization_3')(conv3)
    relu3 = ReLU(name='Decoder_ReLU_3')(bn3)

    # Fourth hidden layer
    conv4 = Conv1D(filters=256, kernel_size=1, name='Decoder_Conv1D_4')(relu3)
    bn4 = BatchNormalization(name='Decoder_BatchNormalization_4')(conv4)
    relu4 = ReLU(name='Decoder_ReLU_4')(bn4)

    # Fifth hidden layer
    conv5 = Conv1D(filters=1024, kernel_size=1, name='Decoder_Conv1D_5')(relu4)
    bn5 = BatchNormalization(name='Decoder_BatchNormalization_5')(conv5)
    relu5 = ReLU(name='Decoder_ReLU_5')(bn5)

    # Global Features (Holistic landmark features)
    global_features = MaxPool1D(pool_size=num_classes, name='Decoder_MaxPool1D')(relu5)

    # Regressing pose parameters
    conv6 = Conv1D(filters=12, kernel_size=1, name='Decoder_Conv1D_6')(global_features)
    bn6 = BatchNormalization(name='Decoder_BatchNormalization_6')(conv6)
    pose_3DMM = ReLU(name='Decoder_ReLU_6')(bn6)

    # Regressing expression parameters
    conv7 = Conv1D(filters=10, kernel_size=1, name='Decoder_Conv1D_7')(global_features)
    bn7 = BatchNormalization(name='Decoder_BatchNormalization_7')(conv7)
    alpha_exp = ReLU(name='Decoder_ReLU_7')(bn7)

    # Regressing shape parameters
    conv8 = Conv1D(filters=40, kernel_size=1, name='Decoder_Conv1D_8')(global_features)
    bn8 = BatchNormalization(name='Decoder_BatchNormalization_8')(conv8)
    alpha_shp = ReLU(name='Decoder_ReLU_8')(bn8)

    model = Model(inputs=[Lr], outputs=[pose_3DMM, alpha_exp, alpha_shp], name='Landmarks_to_3DMM')
    return model