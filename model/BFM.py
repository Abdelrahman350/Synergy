import tensorflow as tf
from tensorflow.keras.layers import Layer

class PCA(Layer):
    def __init__(self, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
    
    def build(self):
        self.w_shp = self.parsing_npy('w_shp_sim.npy')
        self.w_exp = self.parsing_npy('w_exp_sim.npy')
        self.w_tex = self.parsing_npy('w_tex_sim.npy')

    def forward(self):
        pass

    def parsing_npy(self, file):
        npy_file = tf.io.read_file(self.pca_dir+file)
        return tf.io.decode_raw(npy_file, tf.float16)