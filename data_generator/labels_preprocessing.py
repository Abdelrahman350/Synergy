import numpy as np
from numpy import sin, cos, arctan2, arcsin
import pickle

def label_loader(image_id, labels):
    pose_3DMM = np.array(labels[image_id]['Pose'])
    alpha_shp = np.ravel(np.array(labels[image_id]['Shape_Para']).T)
    alpha_exp = np.ravel(np.array(labels[image_id]['Exp_Para']).T)
    parameters_3DMM = np.concatenate((pose_3DMM, alpha_shp, alpha_exp), axis=0)
    return parameters_3DMM

def pose_to_3DMM(pose):
    R = eulerAngles_to_RotationMatrix(pose[:3])
    t = np.expand_dims([*pose[3:5], 0], -1)
    T = np.concatenate((R, t), axis=1)
    s = pose[-1]
    pose_3DMM = T.reshape((-1, ))
    pose_3DMM[-1] = s
    return pose_3DMM

# Calculates Rotation Matrix given euler angles.
def eulerAngles_to_RotationMatrix(theta):
    x, y, z = theta[0], theta[1], theta[2]
    Rx = np.array([[1,       0,      0],
                   [0,  cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])

    Ry = np.array([[cos(y), 0, -sin(y)],
                   [     0, 1,       0],
                   [sin(y), 0,  cos(y)]])
                   
    Rz = np.array([[cos(z) , sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0      ,      0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R

def param3DMM_to_pose(pose_3DMM):
    T = pose_3DMM.reshape((3, 4))
    R = T[:, 0:3]
    return rotationMatrix_to_EulerAngles(R)
    
def rotationMatrix_to_EulerAngles(R):
    ''' compute three Euler angles from a Rotation Matrix. 
        Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        pitch, yaw, roll
    '''
    if R[0, 2] != 1 and R[0, 2] != -1:
        yaw = -arcsin(R[0, 2])
        pitch = arctan2(R[1, 2]/cos(yaw), R[2, 2]/cos(yaw))
        roll = arctan2(R[0, 1]/cos(yaw), R[0, 0]/cos(yaw))
    else:  # Gimbal lock
        roll = 0  # can be anything
        if R[0, 2] == -1:
            yaw = np.pi / 2
            pitch = roll + arctan2(R[1, 0], R[2, 0])
        else:
            yaw = -np.pi / 2
            pitch = -roll + arctan2(-R[1, 0], -R[2, 0])
    return pitch, yaw, roll

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def resize_landmarks(pt2d, aspect_ratio):
    pt2d[:, 0] = pt2d[:, 0] * aspect_ratio[0]
    pt2d[:, 1] = pt2d[:, 1] * aspect_ratio[1]
    return pt2d

def normalize(parameters_3DMM):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')
    parameters_3DMM = (parameters_3DMM - param_mean) / param_std
    return parameters_3DMM

def denormalize(parameters_3DMM):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')
    parameters_3DMM = parameters_3DMM*param_std + param_mean
    return parameters_3DMM

def parsing_pkl(file):
    pca_dir = '3dmm_data/'
    return pickle.load(open(pca_dir+file, 'rb'))

def normalize_dicts(labels):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')
    for parameters_3DMM in labels.values():
        parameters_3DMM['Pose'] = (parameters_3DMM['Pose'] - param_mean[:12]) / param_std[:12]
        parameters_3DMM['Shape_Para'] = (parameters_3DMM['Shape_Para']\
             - param_mean[12:52]) / param_std[12:52]
        parameters_3DMM['Exp_Para'] = (parameters_3DMM['Exp_Para'] - param_mean[52:]) / param_std[52:]
    return labels