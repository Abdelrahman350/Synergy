import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization 
from tensorflow.keras.layers import ReLU, MaxPool1D, concatenate
from tensorflow.keras import Model

# def MMFA(num_points=68):
#     # Z, alpha_shp, alpha_exp, Lc
#     # Define input layers
#     Lc = Input(shape=(68, 3), name='Landmarks')
#     Z = Input(shape=(1, 1280), name='Z')
#     alpha_exp = Input(shape=(1, 10), name='alpha_exp')
#     alpha_shp = Input(shape=(1, 40), name='alpha_shp')

#     # First hidden layer
#     conv1 = Conv1D(filters=64, kernel_size=1, name='Conv1D_1')(Lc)
#     bn1 = BatchNormalization(name='BatchNormalization_1')(conv1)
#     relu1 = ReLU(name='ReLU_1')(bn1)

#     # Second hidden layer
#     conv2 = Conv1D(filters=64, kernel_size=1, name='Conv1D_2')(relu1)
#     bn2 = BatchNormalization(name='BatchNormalization_2')(conv2)
#     relu2 = ReLU(name='ReLU_2')(bn2)

#     # Low-level point features
#     point_features = relu2

#     # Third hidden layer
#     conv3 = Conv1D(filters=64, kernel_size=1, name='Conv1D_3')(relu2)
#     bn3 = BatchNormalization(name='BatchNormalization_3')(conv3)
#     relu3 = ReLU(name='ReLU_3')(bn3)

#     # Fourth hidden layer
#     conv4 = Conv1D(filters=128, kernel_size=1, name='Conv1D_4')(relu3)
#     bn4 = BatchNormalization(name='BatchNormalization_4')(conv4)
#     relu4 = ReLU(name='ReLU_4')(bn4)

#     # Fifth hidden layer
#     conv5 = Conv1D(filters=1024, kernel_size=1, name='Conv1D_5')(relu4)
#     bn5 = BatchNormalization(name='BatchNormalization_5')(conv5)
#     relu5 = ReLU(name='ReLU_5')(bn5)

#     # Global Features (Holistic landmark features)
#     global_features = MaxPool1D(pool_size=num_points, name='MaxPool1D')(relu5)
    
#     # Aggregate point features and global features
#     concate = concatenate([global_features, Z, alpha_exp, alpha_shp], name='Concate_Global_Features')
#     concate_repeat = tf.tile(concate, (1, num_points, 1), name='Repeat_Global_Features')
#     aggregate = concatenate([point_features, concate_repeat])
    
#     # Sixth layer
#     conv6 = Conv1D(filters=512, kernel_size=1, name='Conv1D_6')(aggregate)
#     bn6 = BatchNormalization(name='BatchNormalization_6')(conv6)
#     relu6 = ReLU(name='ReLU_6')(bn6)

#     # Seventh layer
#     conv7 = Conv1D(filters=256, kernel_size=1, name='Conv1D_7')(relu6)
#     bn7 = BatchNormalization(name='BatchNormalization_7')(conv7)
#     relu7 = ReLU(name='ReLU_7')(bn7)

#     # Eighth layer
#     conv8 = Conv1D(filters=128, kernel_size=1, name='Conv1D_8')(relu7)
#     bn8 = BatchNormalization(name='BatchNormalization_8')(conv8)
#     relu8 = ReLU(name='ReLU_8')(bn8)

#     # Ninth layer
#     conv9 = Conv1D(filters=3, kernel_size=1, name='Conv1D_9')(relu8)
#     bn9 = BatchNormalization(name='BatchNormalization_9')(conv9)
#     Lr = ReLU(name='Lr')(bn9)

#     model = Model(inputs=[Lc, Z, alpha_exp, alpha_shp], outputs=[Lr], name='MFFA')
#     return model

class MMFA(tf.keras.Model):
    def __init__(self, num_points=68, **kwargs):
        super(MMFA, self).__init__(**kwargs)
        self.num_points = num_points
        self.conv1 = Conv1D(filters=64, kernel_size=1, name='Conv1D_1')
        self.bn1 = BatchNormalization(name='BatchNormalization_1')
        self.relu1 = ReLU(name='ReLU_1')

        self.conv2 = Conv1D(filters=64, kernel_size=1, name='Conv1D_2')
        self.bn2 = BatchNormalization(name='BatchNormalization_2')
        self.relu2 = ReLU(name='ReLU_2')

        self.conv3 = Conv1D(filters=64, kernel_size=1, name='Conv1D_3')
        self.bn3 = BatchNormalization(name='BatchNormalization_3')
        self.relu3 = ReLU(name='ReLU_3')

        # Fourth hidden layer
        self.conv4 = Conv1D(filters=128, kernel_size=1, name='Conv1D_4')
        self.bn4 = BatchNormalization(name='BatchNormalization_4')
        self.relu4 = ReLU(name='ReLU_4')

        # Fifth hidden layer
        self.conv5 = Conv1D(filters=1024, kernel_size=1, name='Conv1D_5')
        self.bn5 = BatchNormalization(name='BatchNormalization_5')
        self.relu5 = ReLU(name='ReLU_5')

        # Global Features (Holistic landmark features)
        self.global_features = MaxPool1D(pool_size=num_points, name='MaxPool1D')
        
        # Sixth layer
        self.conv6 = Conv1D(filters=512, kernel_size=1, name='Conv1D_6')
        self.bn6 = BatchNormalization(name='BatchNormalization_6')
        self.relu6 = ReLU(name='ReLU_6')

        # Seventh layer
        self.conv7 = Conv1D(filters=256, kernel_size=1, name='Conv1D_7')
        self.bn7 = BatchNormalization(name='BatchNormalization_7')
        self.relu7 = ReLU(name='ReLU_7')

        # Eighth layer
        self.conv8 = Conv1D(filters=128, kernel_size=1, name='Conv1D_8')
        self.bn8 = BatchNormalization(name='BatchNormalization_8')
        self.relu8 = ReLU(name='ReLU_8')

        # Ninth layer
        self.conv9 = Conv1D(filters=3, kernel_size=1, name='Conv1D_9')
        self.bn9 = BatchNormalization(name='BatchNormalization_9')
        self.Lr = ReLU(name='Lr')

    def call(self, Lc, Z, alpha_exp, alpha_shp):
        # First hidden layer
        X = self.conv1(Lc)
        X = self.bn1(X)
        X = self.relu1(X)

        # Second hidden layer
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)

        # Low-level point features
        point_features = X

        # Third hidden layer
        X = self.conv3(X)
        X = self.bn3(X)
        X = self.relu3(X)

        # Fourth hidden layer
        X = self.conv4(X)
        X = self.bn4(X)
        X = self.relu4(X)

        # Fifth hidden layer
        X = self.conv5(X)
        X = self.bn5(X)
        X = self.relu5(X)

        # Global Features (Holistic landmark features)
        global_features = self.global_features(X)
        
        # Aggregate point features and global features
        concate = concatenate([global_features, Z, alpha_exp, alpha_shp], name='Concate_Global_Features')
        concate_repeat = tf.tile(concate, (1, self.num_points, 1), name='Repeat_Global_Features')
        aggregate = concatenate([point_features, concate_repeat])
        
        # Sixth layer
        X = self.conv6(aggregate)
        X = self.bn6(X)
        X = self.relu6(X)

        # Seventh layer
        X = self.conv7(X)
        X = self.bn7(X)
        X = self.relu7(X)

        # Eighth layer
        X = self.conv8(X)
        X = self.bn8(X)
        X = self.relu8(X)

        # Ninth layer
        X = self.conv9(X)
        X = self.bn9(X)
        Lr = self.Lr(X)
        return Lr