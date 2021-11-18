import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class PCA(Layer):
    def __init__(self, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.w_shp = 0
        self.w_exp = 0
        self.w_tex = 0
    
    def build(self):
        self.w_shp = self.parsing_npy('w_shp_sim.npy')
        self.w_exp = self.parsing_npy('w_exp_sim.npy')
        self.w_tex = self.parsing_npy('w_tex_sim.npy')

    def forward(self):
        pass

    def parsing_npy(self, file):
        npy_array = np.load(self.pca_dir+file)
        return tf.convert_to_tensor(npy_array, dtype=tf.float32)
