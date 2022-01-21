import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization 
from tensorflow.keras.layers import ReLU, MaxPool1D, concatenate
from tensorflow.keras import Model
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import L1L2

class MAFA(Model):
    def __init__(self, num_points=68, **kwargs):
        super(MAFA, self).__init__(**kwargs, name='MAFA')
        self.num_points = num_points
        self.conv1 = Conv1D(filters=64, kernel_size=1, name='Encoder_Conv1D_1',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn1 = BatchNormalization(name='Encoder_BatchNormalization_1')
        self.relu1 = ReLU(name='Encoder_ReLU_1')

        self.conv2 = Conv1D(filters=64, kernel_size=1, name='Encoder_Conv1D_2',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn2 = BatchNormalization(name='Encoder_BatchNormalization_2')
        self.relu2 = ReLU(name='Encoder_ReLU_2')

        self.conv3 = Conv1D(filters=64, kernel_size=1, name='Encoder_Conv1D_3',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn3 = BatchNormalization(name='Encoder_BatchNormalization_3')
        self.relu3 = ReLU(name='Encoder_ReLU_3')

        # Fourth hidden layer
        self.conv4 = Conv1D(filters=128, kernel_size=1, name='Encoder_Conv1D_4',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn4 = BatchNormalization(name='Encoder_BatchNormalization_4')
        self.relu4 = ReLU(name='Encoder_ReLU_4')

        # Fifth hidden layer
        self.conv5 = Conv1D(filters=1024, kernel_size=1, name='Encoder_Conv1D_5',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn5 = BatchNormalization(name='Encoder_BatchNormalization_5')
        self.relu5 = ReLU(name='Encoder_ReLU_5')

        # Global Features (Holistic landmark features)
        self.global_features = MaxPool1D(pool_size=num_points, name='Encoder_MaxPool1D')
        
        # Sixth layer
        self.conv6 = Conv1D(filters=512, kernel_size=1, name='Encoder_Conv1D_6',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn6 = BatchNormalization(name='Encoder_BatchNormalization_6')
        self.relu6 = ReLU(name='Encoder_ReLU_6')

        # Seventh layer
        self.conv7 = Conv1D(filters=256, kernel_size=1, name='Encoder_Conv1D_7',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn7 = BatchNormalization(name='Encoder_BatchNormalization_7')
        self.relu7 = ReLU(name='Encoder_ReLU_7')

        # Eighth layer
        self.conv8 = Conv1D(filters=128, kernel_size=1, name='Encoder_Conv1D_8',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn8 = BatchNormalization(name='Encoder_BatchNormalization_8')
        self.relu8 = ReLU(name='Encoder_ReLU_8')

        # Ninth layer
        self.conv9 = Conv1D(filters=3, kernel_size=1, name='Encoder_Conv1D_9',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal(),\
                  kernel_regularizer=L1L2(0.2,0.5), bias_regularizer=L1L2(0.2,0.5))
        self.bn9 = BatchNormalization(name='Encoder_BatchNormalization_9')
        self.Lr = ReLU(name='Encoder_Lr')

    def call(self, Lc, Z, alpha_shp, alpha_exp):
        Z = tf.expand_dims(Z, 1, name='Expanding_Z')
        alpha_shp = tf.expand_dims(alpha_shp, 1, name='Expanding_alphaShp')
        alpha_exp = tf.expand_dims(alpha_exp, 1, name='Expanding_alphaExp')

        # First hidden layer
        X = self.conv1(Lc)
        X = self.bn1(X)
        X = self.relu1(X)

        # Second hidden layer
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)

        # Low-level point features
        point_features = tf.identity(X, name='Point_Features')

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
        concate = concatenate([global_features,\
             Z, alpha_shp, alpha_exp], axis=-1, name='Encoder_Concate_Global_Features')
        concate_repeat = tf.tile(concate,\
             (1, self.num_points, 1), name='Encoder_Repeat_Global_Features')
        aggregate = concatenate([point_features, concate_repeat], name='Encoder_aggregate')
        
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

    def model(self):
        Lc = Input(shape=(68, 3), name='Landmarks')
        Z = Input(shape=(1280,), name='Z')
        alpha_shp = Input(shape=(40,), name='alpha_shp')
        alpha_exp = Input(shape=(10,), name='alpha_exp')
        return Model(inputs=[Lc, Z, alpha_shp, alpha_exp],\
             outputs=self.call(Lc, Z, alpha_shp, alpha_exp))
