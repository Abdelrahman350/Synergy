import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pickle

class PCA(Layer):
    def __init__(self, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.w_shp = 0
        self.w_exp = 0
        self.w_tex = 0
        self.u_shp = 0
        self.u_exp = 0
        self.u_tex = 0
        self.keypoints = 0
    
    def build(self):
        self.w_shp = self.parsing_npy('w_shp_sim.npy')
        self.w_exp = self.parsing_npy('w_exp_sim.npy')
        self.w_tex = self.parsing_npy('w_tex_sim.npy')
        self.u_shp = self.parsing_npy('u_shp_sim.npy')
        self.u_exp = self.parsing_npy('u_exp_sim.npy')
        self.u_tex = self.parsing_npy('u_tex_sim.npy')
        self.keypoints = self.parsing_npy('keypoints_sim.npy')
        self.param_mean = self.parsing_pkl('param_whitening.pkl').get('param_mean')
        self.param_std = self.parsing_pkl('param_whitening.pkl').get('param_std')

    def forward(self):
        pass

    def parsing_npy(self, file):
        npy_array = np.load(self.pca_dir+file)
        return tf.convert_to_tensor(npy_array, dtype=tf.float32)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))