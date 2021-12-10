import scipy.io as sio
import numpy as np

def get_pose_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0].tolist()
    pre_pose_params.pop(5)
    # Get [pitch, yaw, roll, tdx, tdy, scale factor]
    pose_params = np.array(pre_pose_params)
    return pose_params

def get_Exp_Para_from_mat(mat_path):
    # Get Exp_Para landmarks
    mat = sio.loadmat(mat_path)
    Exp_Para = np.ravel(mat['Exp_Para'][0:10])
    return Exp_Para

def get_Shape_Para_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    Exp_Para = np.ravel(mat['Shape_Para'][0:40])
    return Exp_Para