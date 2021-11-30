from data_generator.labels_preprocessing import *
import tensorflow as tf
from tensorflow.keras.layers import Layer, Reshape
import numpy as np
import pickle

class PCA(Layer):
    def __init__(self, input_shape=(224, 224, 3),\
         num_landmarks=68, pca_dir = '3dmm_data/', **kwargs):
        super(PCA, self).__init__(**kwargs)
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.height = 450
        self.u_base = 0
        self.w_exp_base = 0
        self.w_shp_base = 0
        self.aspect_ratio = tf.expand_dims(
            tf.constant([input_shape[0]/450.0, input_shape[1]/450.0, 1]),\
                 0, name='aspect_ratio')

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
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
        self.u_base = self.convert_npy_to_tensor(u[keypoints], 'u_base')
        self.w_exp_base = self.convert_npy_to_tensor(w_exp[keypoints], 'w_exp_base')
        self.w_shp_base = self.convert_npy_to_tensor(w_shp[keypoints], 'w_shp_base')
        self.T = tf.Variable(tf.zeros((self.batch_size, 4, 4)),\
             name='Transformation_Matrix', trainable=False, validate_shape=True)
        batch_diag = tf.repeat(tf.expand_dims(tf.linalg.diag([1.0, 1.0, 1.0, 1.0]), 0), self.batch_size, 0)
        self.M = tf.Variable(batch_diag, name='Move_Matrix', trainable=False)
        # revert height
        self.H = tf.Variable(batch_diag, name='Height_Matrix', trainable=False)
        self.H[:, 1, 1].assign(-tf.ones((self.batch_size, )))
        self.H[:, 1, 3].assign(self.height*tf.ones((self.batch_size, )))
        self.reshape_vertices = Reshape((self.num_landmarks, 3))
        self.reshape_pose = Reshape((3, 4))
        self.reshape_scale = Reshape((1, 1))
        super(PCA, self).build(batch_input_shape)

    def call(self, pose_3DMM, alpha_exp, alpha_shp):
        alpha_exp = tf.expand_dims(alpha_exp, -1)
        alpha_shp = tf.expand_dims(alpha_shp, -1)
        pose_3DMM = tf.cast(pose_3DMM, tf.float32)
        alpha_exp = tf.cast(alpha_exp, tf.float32)
        alpha_shp = tf.cast(alpha_shp, tf.float32)

        vertices = tf.add(self.u_base,\
             tf.add(tf.matmul(self.w_exp_base, alpha_exp, name='1st_Matmul'),\
             tf.matmul(self.w_shp_base, alpha_shp, name='2nd_Matmul'), name='Inner_Add'),\
                  name='Outer_Add')
        vertices = self.reshape_vertices(vertices)
        T_bfm = self.transform_matrix(pose_3DMM)
        temp_ones_vec = tf.ones((self.batch_size, self.num_landmarks, 1), name='1st_Ones')
        homo_vertices = tf.concat((vertices, temp_ones_vec), axis=-1, name='1st_Concat')
        image_vertices = tf.matmul(homo_vertices, T_bfm, transpose_b=True, name='3rd_Matmul')[:, :, 0:3]
        image_vertices_resized = self.resize_landmarks(image_vertices)
        return image_vertices_resized

    def get_config(self):
        base_config = super(PCA, self).get_config()
        return {**base_config, 
        "num_landmarks": self.num_landmarks,
        "pca_dir": self.pca_dir,
        "height": self.height,
        "u_base": self.u_base,
        "w_exp_base": self.w_exp_base,
        "w_shp_base": self.w_shp_base,
        "aspect_ratio": self.aspect_ratio}

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array, name):
        return tf.constant(npy_array, dtype=tf.float32, name=name)
    
    def transform_matrix(self, pose_3DMM):
        """
        :pose_3DMM : [12]
        :return: 4x4 transmatrix
        """
        R, t = self.pose_3DMM_to_sRt(pose_3DMM)
        self.T[:, 0:3, 0:3].assign(R)
        self.T[:, 3, 3].assign(tf.ones((self.batch_size, )))
        # offset move
        self.M[:, 0:3, 3].assign(tf.cast(t, tf.float32))
        self.T = tf.matmul(self.M, self.T)

        self.T = tf.matmul(self.H, self.T)
        return tf.cast(self.T, tf.float32)
    
    def pose_3DMM_to_sRt(self, pose_3DMM):
        T = self.reshape_pose(pose_3DMM)
        R = T[:, :, 0:3]
        t = T[:, :, -1]
        s = tf.expand_dims(t[:, -1], -1)
        s = self.reshape_scale(s)
        diags = tf.expand_dims(tf.linalg.diag([1.0, 1.0, 1.0]), 0)
        diags_repeated = tf.repeat(diags, self.batch_size, 0)
        s = s*diags_repeated
        R = tf.matmul(R, s)
        zero = tf.linalg.diag([1.0, 1.0, 0.0])
        t = tf.matmul(t, zero)
        return R, t
    
    def resize_landmarks(self, pt2d):
        return pt2d*self.aspect_ratio