import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization 
from tensorflow.keras.layers import ReLU, MaxPool1D, concatenate
from tensorflow.keras import Model

def MMFA(num_points=68):
    # Z, alpha_shp, alpha_exp, Lc
    # Define input layers
    Lc = Input(shape=(68, 3), name='Landmarks')
    Z = Input(shape=(1, 1280), name='Z')
    alpha_exp = Input(shape=(1, 10), name='alpha_exp')
    alpha_shp = Input(shape=(1, 40), name='alpha_shp')

    # First hidden layer
    conv1 = Conv1D(filters=64, kernel_size=1, name='Conv1D_1')(Lc)
    bn1 = BatchNormalization(name='BatchNormalization_1')(conv1)
    relu1 = ReLU(name='ReLU_1')(bn1)

    # Second hidden layer
    conv2 = Conv1D(filters=64, kernel_size=1, name='Conv1D_2')(relu1)
    bn2 = BatchNormalization(name='BatchNormalization_2')(conv2)
    relu2 = ReLU(name='ReLU_2')(bn2)

    # Low-level point features
    point_features = relu2

    # Third hidden layer
    conv3 = Conv1D(filters=64, kernel_size=1, name='Conv1D_3')(relu2)
    bn3 = BatchNormalization(name='BatchNormalization_3')(conv3)
    relu3 = ReLU(name='ReLU_3')(bn3)

    # Fourth hidden layer
    conv4 = Conv1D(filters=128, kernel_size=1, name='Conv1D_4')(relu3)
    bn4 = BatchNormalization(name='BatchNormalization_4')(conv4)
    relu4 = ReLU(name='ReLU_4')(bn4)

    # Fifth hidden layer
    conv5 = Conv1D(filters=1024, kernel_size=1, name='Conv1D_5')(relu4)
    bn5 = BatchNormalization(name='BatchNormalization_5')(conv5)
    relu5 = ReLU(name='ReLU_5')(bn5)

    # Global Features (Holistic landmark features)
    global_features = MaxPool1D(pool_size=num_points, name='MaxPool1D')(relu5)
    
    # Aggregate point features and global features
    concate = concatenate([global_features, Z, alpha_exp, alpha_shp], name='Concate_Global_Features')
    concate_repeat = tf.tile(concate, (1, num_points, 1), name='Repeat_Global_Features')
    aggregate = concatenate([point_features, concate_repeat])
    
    # Sixth layer
    conv6 = Conv1D(filters=512, kernel_size=1, name='Conv1D_6')(aggregate)
    bn6 = BatchNormalization(name='BatchNormalization_6')(conv6)
    relu6 = ReLU(name='ReLU_6')(bn6)

    # Seventh layer
    conv7 = Conv1D(filters=256, kernel_size=1, name='Conv1D_7')(relu6)
    bn7 = BatchNormalization(name='BatchNormalization_7')(conv7)
    relu7 = ReLU(name='ReLU_7')(bn7)

    # Eighth layer
    conv8 = Conv1D(filters=128, kernel_size=1, name='Conv1D_8')(relu7)
    bn8 = BatchNormalization(name='BatchNormalization_8')(conv8)
    relu8 = ReLU(name='ReLU_8')(bn8)

    # Ninth layer
    conv9 = Conv1D(filters=3, kernel_size=1, name='Conv1D_9')(relu8)
    bn9 = BatchNormalization(name='BatchNormalization_9')(conv9)
    Lr = ReLU(name='Lr')(bn9)

    model = Model(inputs=[Lc, Z, alpha_exp, alpha_shp], outputs=[Lr], name='MFFA')
    return model