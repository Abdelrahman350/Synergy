from data_generator.preprocessing_labels import eulerAngles_to_RotationMatrix, rotationMatrix_to_EulerAngles
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import pickle

class PCA(Layer):
    def __init__(self, height, num_landmarks=68, pca_dir = '3dmm_data/'):
        super(PCA, self).__init__()
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.height = height
        self.u_base = 0
        self.w_exp_base = 0
        self.w_shp_base = 0
    
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
        vertices = tf.reshape(vertices, (int(len(vertices)/3), 3))
        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]

        T_bfm = self.transform_matrix(s, angles, t, self.height)
        temp_ones_vec = tf.ones((len(vertices), 1))
        homo_vertices = tf.concat((vertices, temp_ones_vec), axis=-1)
        image_vertices = tf.matmul(homo_vertices, tf.transpose(T_bfm))[:, 0:3]
        return image_vertices

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array):
        return tf.Variable(npy_array, dtype=tf.float32, trainable=False)
    
    def transform_matrix(self, s, angles, t, height):
        """
        :param s: scale
        :param angles: [3] rad
        :param t: [3]
        :return: 4x4 transmatrix
        """
        R = self.eulerAngles_to_RotationMatrix(angles)
        R = tf.cast(R, tf.float32)
        T = tf.Variable(tf.zeros((4, 4)))
        T = T[0:3, 0:3].assign(R)
        T = T[3, 3].assign(1.0)
        # scale
        S = tf.linalg.diag([s, s, s, 1.0])
        T = tf.matmul(S, T)
        # offset move
        M = tf.Variable(tf.linalg.diag([1.0, 1.0, 1.0, 1.0]))
        M = M[0:3, 3].assign(tf.cast(t, tf.float32))
        T = tf.matmul(M, T)
        # revert height
        H = tf.Variable(tf.linalg.diag([1.0, 1.0, 1.0, 1.0]))
        H = H[1, 1].assign(-1.0)
        H = H[1, 3].assign(height)
        T = tf.matmul(H, T)
        return tf.cast(T, tf.float32)

    def eulerAngles_to_RotationMatrix(self, theta):
        x, y, z = theta[0], theta[1], theta[2]
        Rx = tf.Variable([[1,          0,         0],
                          [0,  tf.cos(x), tf.sin(x)],
                          [0, -tf.sin(x), tf.cos(x)]], trainable=False)

        Ry = tf.Variable([[tf.cos(y), 0, -tf.sin(y)],
                          [        0, 1,          0],
                          [tf.sin(y), 0,  tf.cos(y)]], trainable=False)
                    
        Rz = tf.Variable([[tf.cos(z) , tf.sin(z), 0],
                          [-tf.sin(z), tf.cos(z), 0],
                          [0         ,         0, 1]], trainable=False)
        R = tf.matmul(Rx, tf.matmul(Ry, Rz))
        return R