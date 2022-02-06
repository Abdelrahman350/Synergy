import pickle
import numpy as np
from numpy import expand_dims, reshape, matmul, squeeze, multiply, transpose

def predict_lmks(param_3DMM, roi_box):
    lmks = squeeze(PCA(input_shape=(450, 450, 3))(param_3DMM), 0)
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 450
    scale_y = (ey - sy) / 450
    lmks[:, 0] = lmks[:, 0] * scale_x + sx
    lmks[:, 1] = lmks[:, 1] * scale_y + sy
    s = (scale_x + scale_y) / 2
    lmks[:, 2] *= s
    return lmks

class PCA(object):
    def __init__(self, input_shape=(128, 128, 3), height=450.0,\
        num_landmarks=68, pca_dir = '3dmm_data/', **kwargs):
        self.num_landmarks = num_landmarks
        self.pca_dir = pca_dir
        self.height = height
        self.u_base = 0
        self.w_shp_base = 0
        self.w_exp_base = 0
        self.aspect_ratio = expand_dims(
            np.array([input_shape[0]/self.height,input_shape[1]/self.height,1]), 0)
        w_shp = self.parsing_npy('w_shp_sim.npy')
        w_exp = self.parsing_npy('w_exp_sim.npy')
        w_tex = self.parsing_npy('w_tex_sim.npy')
        u_shp = self.parsing_npy('u_shp.npy')
        u_exp = self.parsing_npy('u_exp.npy')
        u_tex = self.parsing_npy('u_tex.npy')
        keypoints = self.parsing_npy('keypoints_sim.npy')
        self.param_mean = self.parsing_pkl('param_300W_LP.pkl').get('param_mean')
        self.param_std = self.parsing_pkl('param_300W_LP.pkl').get('param_std')
        u = u_shp + u_exp
        self.u_base = u[keypoints]
        self.w_shp_base = w_shp[keypoints]
        self.w_exp_base = w_exp[keypoints]        

    def __call__(self, Param_3D):
        Param_3D = self.param_std*Param_3D + self.param_mean
        pose_3DMM, alpha_shp, alpha_exp = Param_3D[:, :12], Param_3D[:, 12:52], Param_3D[:, 52:]
        alpha_exp = expand_dims(alpha_exp, -1)
        alpha_shp = expand_dims(alpha_shp, -1)
        pose_3DMM = pose_3DMM.astype(np.float)
        alpha_shp = alpha_shp.astype(np.float)
        alpha_exp = alpha_exp.astype(np.float)

        vertices = self.u_base + matmul(self.w_exp_base, alpha_exp) + matmul(self.w_shp_base, alpha_shp)
        vertices = reshape(vertices, (self.num_landmarks, 3))
        T, t = self.transform_matrix(pose_3DMM)
        vertices = matmul(vertices, transpose(T, axes=(0, 2, 1))) + expand_dims(t, 1)
        vertices = self.resize_landmarks(vertices)
        return vertices

    def parsing_npy(self, file):
        return np.load(self.pca_dir+file)
    
    def parsing_pkl(self, file):
        return pickle.load(open(self.pca_dir+file, 'rb'))
    
    def transform_matrix(self, pose_3DMM):
        """
        :pose_3DMM : [12]
        :return: 4x4 transmatrix
        """
        s, R, t = self.pose_3DMM_to_sRt(pose_3DMM)
        # Zeroing the tz
        zero = np.diag([1.0, 1.0, 0.0])
        t = matmul(t, zero) + np.array([0.0, 0.0, 1.0])
        # Convert ty ----> (Height_image - ty)
        H_t = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, self.height, 0.0]])
        t = matmul(t, H_t)
        # Negative 2nd row in R
        H_R = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        T = H_R * multiply(reshape(s, (1, 1)), R)
        return T, t
    
    def pose_3DMM_to_sRt(self, pose_3DMM):
        T = reshape(pose_3DMM, (1, 3, 4))
        R = T[:, :, 0:3]
        t = T[:, :, -1]
        s = t[:, -1]
        return s, R, t
    
    def resize_landmarks(self, pt2d):
        return pt2d*self.aspect_ratio