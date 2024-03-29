import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization 
from tensorflow.keras.layers import ReLU, GlobalMaxPool1D, Reshape
from tensorflow.keras import Model
from tensorflow.keras.initializers import GlorotNormal

class Landmarks_to_3DMM(Model):
    def __init__(self, num_points=68, **kwargs):
        super(Landmarks_to_3DMM, self).__init__(**kwargs, name="Landmarks_to_3DMM")
        self.num_points = num_points
        # First hidden layer
        self.conv1 = Conv1D(filters=64, kernel_size=1, name='Decoder_Conv1D_1',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn1 = BatchNormalization(name='Decoder_BatchNormalization_1')
        self.relu1 = ReLU(name='Decoder_ReLU_1')
        
        # Second hidden layer
        self.conv2 = Conv1D(filters=64, kernel_size=1, name='Decoder_Conv1D_2',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn2 = BatchNormalization(name='Decoder_BatchNormalization_2')
        self.relu2 = ReLU(name='Decoder_ReLU_2')
        
        # Third hidden layer
        self.conv3 = Conv1D(filters=64, kernel_size=1, name='Decoder_Conv1D_3',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn3 = BatchNormalization(name='Decoder_BatchNormalization_3')
        self.relu3 = ReLU(name='Decoder_ReLU_3')
        
        # Fourth hidden layer
        self.conv4 = Conv1D(filters=128, kernel_size=1, name='Decoder_Conv1D_4',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn4 = BatchNormalization(name='Decoder_BatchNormalization_4')
        self.relu4 = ReLU(name='Decoder_ReLU_4')
        
        # Fifth hidden layer
        self.conv5 = Conv1D(filters=1024, kernel_size=1, name='Decoder_Conv1D_5',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn5 = BatchNormalization(name='Decoder_BatchNormalization_5')
        self.relu5 = ReLU(name='Decoder_ReLU_5')
        
        # Global Features (Holistic landmark features)
        self.maxPool = GlobalMaxPool1D(name='Decoder_MaxPool1D')
        self.reshape = Reshape((1, -1))

        # Regressing pose parameters
        self.conv6 = Conv1D(filters=12, kernel_size=1, name='Decoder_Conv1D_6',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn6 = BatchNormalization(name='Decoder_BatchNormalization_6')
        self.relu_6 = ReLU(name='Decoder_ReLU_6')

        # Regressing shape parameters
        self.conv7 = Conv1D(filters=40, kernel_size=1, name='Decoder_Conv1D_7',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn7 = BatchNormalization(name='Decoder_BatchNormalization_7')
        self.relu_7 = ReLU(name='Decoder_ReLU_7')

        # Regressing expression parameters
        self.conv8 = Conv1D(filters=10, kernel_size=1, name='Decoder_Conv1D_8',\
             kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal())
        self.bn8 = BatchNormalization(name='Decoder_BatchNormalization_8')
        self.relu_8 = ReLU(name='Decoder_ReLU_8')

    def call(self, Lr):
        # First hidden layer
        X = self.conv1(Lr)
        X = self.bn1(X)
        X = self.relu1(X)

        # Second hidden layer
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)

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
        X = self.maxPool(X)
        global_features = self.reshape(X)

        # Regressing pose parameters
        X = self.conv6(global_features)
        X = self.bn6(X)
        pose_3DMM = self.relu_6(X)

        # Regressing shape parameters
        X = self.conv7(global_features)
        X = self.bn7(X)
        alpha_shp = self.relu_7(X)

        # Regressing expression parameters
        X = self.conv8(global_features)
        X = self.bn8(X)
        alpha_exp = self.relu_8(X)
        
        pose_3DMM = tf.squeeze(pose_3DMM, 1, name="Squeezing_pose3DMM")
        alpha_shp = tf.squeeze(alpha_shp, 1, name="Squeezing_alpha_shp")
        alpha_exp = tf.squeeze(alpha_exp, 1, name="Squeezing_alpha_exp")

        Param_3D_hat = tf.concat((pose_3DMM, alpha_shp, alpha_exp), axis=-1)
        return Param_3D_hat

    def model(self):
        Lr = Input(shape=(self.num_points, 3), name='Refined_Landmarks')
        return Model(inputs=Lr, outputs=self.call(Lr))