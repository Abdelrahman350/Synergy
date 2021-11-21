from data_generator.preprocessing_labels import eulerAngles_to_RotationMatrix, rotationMatrix_to_EulerAngles
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pickle

class PCA(Layer):
    def __init__(self, height, num_landmarks=68, pca_dir = '3dmm_data/', **kwargs):
        super(PCA, self).__init__(**kwargs)
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.height = height
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

    def call(self, pose_3DMM, alpha_exp, alpha_shp):
        alpha_exp = tf.expand_dims(alpha_exp, -1)
        alpha_shp = tf.expand_dims(alpha_shp, -1)
        pose_3DMM = tf.cast(pose_3DMM, tf.float32)
        alpha_exp = tf.cast(alpha_exp, tf.float32)
        alpha_shp = tf.cast(alpha_shp, tf.float32)

        vertices = tf.add(self.u_base, tf.add(tf.matmul(self.w_exp_base, alpha_exp),\
             tf.matmul(self.w_shp_base, alpha_shp)))
        vertices = tf.reshape(vertices, (tf.shape(vertices)[0], self.num_landmarks, 3))
        T_bfm = self.transform_matrix(pose_3DMM, self.height)
        temp_ones_vec = tf.ones((tf.shape(vertices)[0], tf.shape(vertices)[1], 1))
        homo_vertices = tf.concat((vertices, temp_ones_vec), axis=-1)
        image_vertices = tf.matmul(homo_vertices, tf.transpose(T_bfm))[:, :, 0:3]
        return image_vertices

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array):
        return tf.Variable(npy_array, dtype=tf.float32, trainable=False)
    
    def transform_matrix(self, pose_3DMM, height):
        """
        :pose_3DMM : [12]
        :return: 4x4 transmatrix
        """
        s, R, t = self.pose_3DMM_to_sRt(pose_3DMM)
        T = tf.Variable(lambda: tf.zeros((4, 4)))
        T = T[0:3, 0:3].assign(R)
        T = T[3, 3].assign(1.0)
        # scale
        S = tf.linalg.diag([s, s, s, 1.0])
        T = tf.matmul(S, T)
        # offset move
        M = tf.Variable(lambda: tf.linalg.diag([1.0, 1.0, 1.0, 1.0]))
        t = tf.reshape(t, [-1])
        M = M[0:3, 3].assign(tf.cast(t, tf.float32))
        T = tf.matmul(M, T)
        # revert height
        H = tf.Variable(lambda: tf.linalg.diag([1.0, 1.0, 1.0, 1.0]))
        H = H[1, 1].assign(-1.0)
        H = H[1, 3].assign(height)
        T = tf.matmul(H, T)
        return tf.cast(T, tf.float32)
    
    def pose_3DMM_to_sRt(self, pose_3DDM):
        T = tf.reshape(pose_3DDM, (3, 4))
        R = T[:, 0:3]
        t = tf.expand_dims(T[:, -1], -1)
        s = tf.reduce_sum(t[-1])
        zero = tf.linalg.diag([1.0, 1.0, 0.0])
        t = tf.matmul(zero, t)
        return s, R, t