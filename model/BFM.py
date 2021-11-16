import tensorflow as tf
from tensorflow.keras.layers import Layer

class PCA(Layer):
    def __init__(self, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
    
    def build(self):
        

    def forward(self):
        pass