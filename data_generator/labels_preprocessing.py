import numpy as np
from numpy import sin, cos, arctan2, arcsin
import pickle

def label_loader(image_id, labels):
    Pose_3DMM = np.array(labels[image_id]['Pose'])
    alpha_shp = np.ravel(np.array(labels[image_id]['Shape_Para']).T)
    alpha_exp = np.ravel(np.array(labels[image_id]['Exp_Para']).T)
    Param_3D = np.concatenate((Pose_3DMM, alpha_shp, alpha_exp), axis=0)
    return Param_3D

def pose_to_3DMM(pose):
    R = eulerAngles_to_RotationMatrix(pose[:3])
    t = np.expand_dims([*pose[3:5], 0], -1)
    T = np.concatenate((R, t), axis=1)
    s = pose[-1]
    Pose_3DMM = T.reshape((-1, ))
    Pose_3DMM[-1] = s
    return Pose_3DMM

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

def P2sRt(P):
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    norm_R1 = np.linalg.norm(R1)
    norm_R2 = np.linalg.norm(R2)
    s = (norm_R1 + norm_R2) / 2.0
    r1 = R1 / norm_R1
    r2 = R2 / norm_R2
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def Param3D_to_Pose(Param_3D):
    Param_3D = np.array(Param_3D[:12])
    P = Param_3D.reshape((3, 4))
    s, R, t3d = P2sRt(P)
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
    theta = [pitch, yaw, roll]
    return theta

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

def normalize_param(Param_3D):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')
    Param_3D = (Param_3D - param_mean) / param_std
    return Param_3D

def denormalize_param(Param_3D):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')[:62]
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')[:62]
    Param_3D = Param_3D*param_std + param_mean
    return Param_3D

def denormalize_DDFA(Param_3D):
    param_mean = parsing_pkl('param_whitening.pkl').get('param_mean')[:62]
    param_std = parsing_pkl('param_whitening.pkl').get('param_std')[:62]
    Param_3D = Param_3D*param_std + param_mean
    return Param_3D

def parsing_pkl(file):
    pca_dir = '3dmm_data/'
    return pickle.load(open(pca_dir+file, 'rb'))

def normalize_dicts(labels):
    param_mean = parsing_pkl('param_300W_LP.pkl').get('param_mean')
    param_std = parsing_pkl('param_300W_LP.pkl').get('param_std')
    for Param_3D in labels.values():
        Param_3D['Pose'] = (Param_3D['Pose'] - param_mean[:12]) / param_std[:12]
        Param_3D['Shape_Para'] = (Param_3D['Shape_Para'] - param_mean[12:52]) / param_std[12:52]
        Param_3D['Exp_Para'] = (Param_3D['Exp_Para'] - param_mean[52:]) / param_std[52:]
    return labels