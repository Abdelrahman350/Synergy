import pickle
import numpy as np
import tensorflow as tf
from tensorflow import expand_dims, cast, constant, add, matmul
from tensorflow.keras.metrics import Metric, MeanAbsoluteError
from tensorflow.keras.layers import Layer, Reshape

class OrientationMAE(Metric):
    def __init__(self, **kwargs):
        super(OrientationMAE, self).__init__(**kwargs)
        self.mae = MeanAbsoluteError()
        self.reshape_pose = Reshape((3, 4))
        self.pitch = self.add_weight("Pitch", initializer='zeros')
        self.yaw = self.add_weight("Yaw", initializer='zeros')
        self.roll = self.add_weight("Roll", initializer='zeros')
        self.count = self.add_weight("Count", initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = self.denormalize(y_true)
        y_pred = self.denormalize(y_pred)
        pitch_true, yaw_true, roll_true = self.param3DMM_to_pose(y_true)
        pitch_pred, yaw_pred, roll_pred = self.param3DMM_to_pose(y_pred)
        self.pitch.assign_add(self.mae(pitch_true, pitch_pred))
        self.yaw.assign_add(self.mae(yaw_true, yaw_pred))
        self.roll.assign_add(self.mae(roll_true, roll_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        pitch = self.pitch / self.count
        yaw = self.yaw / self.count
        roll = self.roll / self.count
        avg = (pitch + yaw + roll) / 3.0
        return pitch, yaw, roll, avg

    def denormalize(self, parameters_3DMM):
        param_mean = self.parsing_pkl('param_300W_LP.pkl').get('param_mean')[:62]
        param_std = self.parsing_pkl('param_300W_LP.pkl').get('param_std')[:62]
        parameters_3DMM = parameters_3DMM*param_std + param_mean
        return parameters_3DMM
    
    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def convert_npy_to_tensor(self, npy_array, name):
        return constant(npy_array, dtype=tf.float32, name=name)
    
    def param3DMM_to_pose(self, pose_3DMM):
        P = self.reshape_pose(pose_3DMM)
        s, R, t3d = self.P2sRt(P)
        return self.rotationMatrix_to_EulerAngles(R)
    
    def P2sRt(self, P):
        t3d = P[:, :, 3]
        R1 = P[:, 0:1, :3]
        R2 = P[:, 1:2, :3]
        norm_R1 = tf.linalg.norm(R1)
        norm_R2 = tf.linalg.norm(R2)
        s = (norm_R1 + norm_R2) / 2.0
        r1 = R1 / norm_R1
        r2 = R2 / norm_R2
        r3 = tf.linalg.cross(r1, r2)
        R = tf.concat((r1, r2, r3), 0)
        return s, R, t3d
    
    def rotationMatrix_to_EulerAngles(self, R):
        ''' compute three Euler angles from a Rotation Matrix. 
            Ref: http://www.gregslabaugh.net/publications/euler.pdf
        Args:
            R: (3,3). rotation matrix
        Returns:
            pitch, yaw, roll
        '''
        if R[:, 0, 2] != 1 and R[:, 0, 2] != -1:
            yaw = -tf.asin(R[:, 0, 2])
            pitch = tf.atan2(R[:, 1, 2]/tf.cos(yaw), R[:, 2, 2]/tf.cos(yaw))
            roll = tf.atan2(R[:, 0, 1]/tf.cos(yaw), R[:, 0, 0]/tf.cos(yaw))
        else:  # Gimbal lock
            roll = 0  # can be anything
            if R[:, 0, 2] == -1:
                yaw = np.pi / 2
                pitch = roll + tf.atan2(R[:, 1, 0], R[:, 2, 0])
            else:
                yaw = -np.pi / 2
                pitch = -roll + tf.atan2(-R[:, 1, 0], -R[:, 2, 0])
        return pitch, yaw, roll