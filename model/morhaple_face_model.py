from data_generator.preprocessing_labels import eulerAngles_to_RotationMatrix, rotationMatrix_to_EulerAngles
import tensorflow as tf
from tensorflow.keras.layers import Layer
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
        self.u_base = self.convert_npy_to_tensor(u[keypoints], 'u_base')
        self.w_exp_base = self.convert_npy_to_tensor(w_exp[keypoints], 'w_exp_base')
        self.w_shp_base = self.convert_npy_to_tensor(w_shp[keypoints], 'w_shp_base')

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
        vertices = tf.reshape(vertices, (tf.shape(vertices)[0], self.num_landmarks, 3),\
             name='1st_Reshape')
        T_bfm = self.transform_matrix(pose_3DMM, self.height)
        temp_ones_vec = tf.ones((tf.shape(vertices)[0], tf.shape(vertices)[1], 1), name='1st_Ones')
        homo_vertices = tf.concat((vertices, temp_ones_vec), axis=-1, name='1st_Concat')
        image_vertices = tf.matmul(homo_vertices, tf.transpose(T_bfm), name='3rd_Matmul')[:, :, 0:3]
        image_vertices_resized = self.resize_landmarks(image_vertices)
        return image_vertices_resized

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array, name):
        return tf.constant(npy_array, dtype=tf.float32, name=name)
    
    def transform_matrix(self, pose_3DMM, height):
        """
        :pose_3DMM : [12]
        :return: 4x4 transmatrix
        """
        s, R, t = self.pose_3DMM_to_sRt(pose_3DMM)
        T = tf.Variable(lambda: tf.zeros((4, 4)), name='Transformation_Matrix')
        T = T[0:3, 0:3].assign(R)
        T = T[3, 3].assign(1.0)
        # scale
        S = tf.linalg.diag([s, s, s, 1.0])
        T = tf.matmul(S, T)
        # offset move
        M = tf.Variable(lambda: tf.linalg.diag([1.0, 1.0, 1.0, 1.0]), name='Move_Matrix')
        t = tf.reshape(t, [-1])
        M = M[0:3, 3].assign(tf.cast(t, tf.float32))
        T = tf.matmul(M, T)
        # revert height
        H = tf.Variable(lambda: tf.linalg.diag([1.0, 1.0, 1.0, 1.0]), name='Height_Matrix')
        H = H[1, 1].assign(-1.0)
        H = H[1, 3].assign(height)
        T = tf.matmul(H, T)
        return tf.cast(T, tf.float32)
    
    def pose_3DMM_to_sRt(self, pose_3DDM):
        T = tf.reshape(pose_3DDM, (tf.shape(pose_3DDM)[0], 3, 4), name='2nd_Reshape')
        R = T[:, :, 0:3]
        t = tf.expand_dims(T[:, :, -1], -1)
        s = tf.reduce_sum(t[-1])
        zero = tf.linalg.diag([1.0, 1.0, 0.0])
        t = tf.matmul(zero, t)
        return s, R, t
    
    def resize_landmarks(self, pt2d):
        return pt2d*self.aspect_ratio