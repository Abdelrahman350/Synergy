import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pickle

class PCA(Layer):
    def __init__(self, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.u_base = 0
        self.w_exp_base = 0
        self.w_shp_base = 0
    
    def build(self):
        w_exp = self.parsing_npy('w_exp_sim.npy')
        w_shp = self.parsing_npy('w_shp_sim.npy')
        w_tex = self.parsing_npy('w_tex_sim.npy')
        u_exp = self.parsing_npy('u_exp.npy')
        u_shp = self.parsing_npy('u_shp.npy')
        u_tex = self.parsing_npy('u_tex.npy')
        keypoints = self.parsing_npy('keypoints_sim.npy')
        param_mean = self.parsing_pkl('param_whitening.pkl').get('param_mean')
        param_std = self.parsing_pkl('param_whitening.pkl').get('param_std')
        u = u_exp + u_shp
        self.u_base = self.convert_npy_to_tensor(u[keypoints])
        self.w_exp_base = self.convert_npy_to_tensor(w_exp[keypoints])
        self.w_shp_base = self.convert_npy_to_tensor(w_shp[keypoints])

    def call(self, pose_para, alpha_exp, alpha_shp):
        vertices = self.u_base + tf.matmul(self.w_exp_base, alpha_exp) +\
             tf.matmul(self.w_shp_base, alpha_shp)
        vertices = gittf.reshape(vertices, (int(len(vertices)/3), 3))
        return vertices

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array):
        return tf.convert_to_tensor(npy_array, dtype=tf.float32)