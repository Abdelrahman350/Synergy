import math 
import numpy as np
from numpy import sin, cos, arctan2, arcsin

def label_loader(image_id, labels):
    pose = np.array(labels[image_id]['pose'])
    Exp_Para = np.array(labels[image_id]['Exp_Para']).T
    Shape_Para = np.array(labels[image_id]['Shape_Para']).T
    pt2d = np.array(labels[image_id]['pt2d']).T
    pose_3DMM = pose_to_3DMM(pose)
    parameters_3DMM = np.concatenate((pose_3DMM, Exp_Para, Shape_Para), axis=1)
    return parameters_3DMM, pt2d

def pose_to_3DMM(pose):
    R = eulerAngles_to_RotationMatrix(pose[:3])
    t = np.expand_dims([*pose[3:5], 0], -1)
    T = np.concatenate((R, t), axis=1)
    f = pose[-1]
    pose_3DDM = T.reshape((1, -1))
    pose_3DDM[0, -1] = f
    return pose_3DDM

# Calculates Rotation Matrix given euler angles.
def eulerAngles_to_RotationMatrix(theta) :
    R_x = np.array([[1,         0,                   0        ],
                    [0,         cos(theta[0]), -sin(theta[0]) ],
                    [0,         sin(theta[0]),  cos(theta[0]) ]
                    ])

    R_y = np.array([[cos(theta[1]),    0,      sin(theta[1])  ],
                    [0,                1,      0              ],
                    [-sin(theta[1]),   0,      cos(theta[1])  ]
                    ])

    R_z = np.array([[cos(theta[2]),   -sin(theta[2]),     0],
                    [sin(theta[2]),    cos(theta[2]),     0],
                    [0,                     0,            1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def label_3DDm_to_pose(label):
    parameters_3DMM = label_to_3DMM(label)
    pose_3DDM = parameters_3DMM[:, 0:12]
    T = pose_3DDM.reshape((3, 4))
    R = T[:, 0:3]
    return rotationMatrix_to_EulerAngles(R)

def label_to_3DMM(label):
    return label[0, 0]

def rotationMatrix_to_EulerAngles(R):
    ''' compute three Euler angles from a Rotation Matrix. 
        Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        pitch, yaw, roll
    '''
    if R[2, 0] != 1 and R[2, 0] != -1:
        yaw = -arcsin(R[2, 0])
        pitch = arctan2(R[2, 1]/cos(yaw), R[2, 2]/cos(yaw))
        roll = arctan2(R[1, 0]/cos(yaw), R[0, 0]/cos(yaw))
    else:  # Gimbal lock
        roll = 0  # can be anything
        if R[2, 0] == -1:
            yaw = np.pi / 2
            pitch = roll + arctan2(R[0, 1], R[0, 2])
        else:
            yaw = -np.pi / 2
            pitch = -roll + arctan2(-R[0, 1], -R[0, 2])
    return pitch, yaw, roll

def label_3DDm_to_pt2d(label):
    pt2d = label_to_pt2d(label)
    return pt2d

def label_to_pt2d(label):
    return label[0, 1]