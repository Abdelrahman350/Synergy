import pickle
import numpy as np
import tensorflow as tf
from tensorflow import expand_dims, constant
from tensorflow.keras.metrics import Metric, mean_absolute_error
from tensorflow.keras.layers import Reshape
from tensorflow import abs, reduce_mean

class OrientationMAE(Metric):
    def __init__(self, mode='avg', **kwargs):
        super(OrientationMAE, self).__init__(**kwargs)
        self.mode = mode
        self.reshape_pose = Reshape((3, 4))
        self.pitch = self.add_weight("Pitch", initializer='zeros')
        self.yaw = self.add_weight("Yaw", initializer='zeros')
        self.roll = self.add_weight("Roll", initializer='zeros')
        self.count = self.add_weight("Count", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.denormalize(y_true)[:, :12]
        y_pred = self.denormalize(y_pred)[:, :12]
        angles_true = self.param3DMM_to_pose(y_true)*180/np.pi
        angles_pred = self.param3DMM_to_pose(y_pred)*180/np.pi
        pitch_metric = reduce_mean(abs(angles_true[0, :] - angles_pred[0, :]))
        yaw_metric = reduce_mean(abs(angles_true[1, :] - angles_pred[1, :]))
        roll_metric = reduce_mean(abs(angles_true[2, :] - angles_pred[2, :]))

        self.pitch.assign_add(pitch_metric)
        self.yaw.assign_add(yaw_metric)
        self.roll.assign_add(roll_metric)
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        pitch = self.pitch / self.count
        yaw = self.yaw / self.count
        roll = self.roll / self.count
        avg = (pitch + yaw + roll) / 3.0

        if self.mode == 'avg':
            return avg
        elif self.mode == 'pitch':
            return pitch
        elif self.mode == 'yaw':
            return yaw
        elif self.mode == 'roll':
            return roll

    def denormalize(self, parameters_3DMM):
        param_mean = self.parsing_pkl('param_300W_LP.pkl').get('param_mean')[:62]
        param_std = self.parsing_pkl('param_300W_LP.pkl').get('param_std')[:62]
        parameters_3DMM = parameters_3DMM*param_std + param_mean
        return parameters_3DMM
    
    def parsing_npy(self, file):
        pca_dir = '3dmm_data/'
        return np.load(pca_dir+file)
    
    def parsing_pkl(self, file):
        pca_dir = '3dmm_data/'
        return pickle.load(open(pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array, name):
        return constant(npy_array, dtype=tf.float32, name=name)
    
    def param3DMM_to_pose(self, pose_3DMM):
        P = self.reshape_pose(pose_3DMM)
        s, R, t3d = self.P2sRt(P)
        return self.rotationMatrix_to_EulerAngles(R)
    
    def P2sRt(self, P):
        t3d = P[:, :, 3]
        R1 = P[:, 0, 0:3]
        R2 = P[:, 1, 0:3]
        norm_R1 = tf.expand_dims(tf.linalg.norm(R1, axis=-1), -1)
        norm_R2 = tf.expand_dims(tf.linalg.norm(R2, axis=-1), -1)
        s = (norm_R1 + norm_R2) / 2.0
        r1 = expand_dims(R1 / norm_R1, 1)
        r2 = expand_dims(R2 / norm_R2, 1)
        r3 = tf.linalg.cross(r1, r2)
        R = tf.concat((r1, r2, r3), 1)
        return s, R, t3d
    
    def rotationMatrix_to_EulerAngles(self, R):
        ''' compute three Euler angles from a Rotation Matrix. 
            Ref: http://www.gregslabaugh.net/publications/euler.pdf
        Args:
            R: (3,3). rotation matrix
        Returns:
            pitch, yaw, roll
        '''
        def branch_one(R):
            yaw = -tf.asin(R[:, 0, 2])
            pitch = tf.atan2(R[:, 1, 2]/tf.cos(yaw), R[:, 2, 2]/tf.cos(yaw))
            roll = tf.atan2(R[:, 0, 1]/tf.cos(yaw), R[:, 0, 0]/tf.cos(yaw))
            return pitch, yaw, roll
        def branch_two(R):
            cond_2 =  R[:, 0, 2] == -1
            angles = tf.where(cond_2, branch_three(R), branch_four(R))
            return angles
        def branch_three(R):
            roll = tf.squeeze(tf.matmul(tf.transpose(tf.matmul(R,\
                 tf.zeros((1, 3, 1))), perm=(0, 2, 1)), tf.zeros((1, 3, 1))), [-1, -2])
            yaw = np.pi / 2 + roll
            pitch = roll + tf.atan2(R[:, 1, 0], R[:, 2, 0])
            return pitch, yaw, roll
        def branch_four(R):
            roll = tf.squeeze(tf.matmul(tf.transpose(tf.matmul(R,\
                 tf.zeros((1, 3, 1))), perm=(0, 2, 1)), tf.zeros((1, 3, 1))), [-1, -2])
            yaw = -np.pi / 2 + roll
            pitch = -roll + tf.atan2(-R[:, 1, 0], -R[:, 2, 0])
            return pitch, yaw, roll
        r1 = R[:, 0, 2] != 1
        r2 = R[:, 0, 2] != -1
        cond_1 =  tf.math.logical_and(r1, r2)
        angles = tf.where(cond_1, branch_one(R), branch_two(R))
        return angles