from model.morhaple_face_model import PCA
from data_generator.labels_preprocessing import label_to_3DMM, label_to_pt2d
import tensorflow as tf
from tensorflow.math import square, reduce_sum
from tensorflow.keras.losses import Loss, Huber

class Synergy_Loss(Loss):
    def __init__(self, lamda_1=0.02, lamda_2=0.05, lamda_3=0.02,\
         lamda_4=0.001, input_shape=(224, 224, 3), **kwargs):
        super(Synergy_Loss, self).__init__(**kwargs)
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.lamda_3 = lamda_3
        self.lamda_4 = lamda_4
        self.huber = Huber()
        self.input_shape = input_shape
        self.pca = PCA(input_shape=self.input_shape)
        
    def call(self, y_gt, y_pred):
        pose_3DMM_gt = y_gt[:, 0:12]
        pose_3DMM = y_pred[:, 0:12]

        alpha_exp_gt = y_gt[:, 12:22]
        alpha_shp_gt = y_gt[:, 22:62]
        alpha_exp = y_pred[:, 12:22]
        alpha_shp = y_pred[:, 22:62]

        L_gt = self.pca(pose_3DMM_gt, alpha_exp_gt, alpha_shp_gt)
        Lr = self.pca(pose_3DMM, alpha_exp, alpha_shp)

        L3DMM = self.L3DMM_loss(pose_3DMM_gt, pose_3DMM)
        L_lmk = self.L_lmk_loss(L_gt, Lr)
        L3DMM_lmk = 0#self.L3DMM_lmk_loss(pose_3DMM_true, pose_3DMM_hat)
        Lg = 0#self.Lg_loss(pose_3DMM, pose_3DMM_hat)
        L_total  = self.lamda_1*L3DMM + self.lamda_2*L_lmk +\
             self.lamda_3*L3DMM_lmk + self.lamda_4*Lg
        return L_total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "lamda_1": self.lamda_1,\
             "lamda_2": self.lamda_2, "lamda_3": self.lamda_3, "lamda_4": self.lamda_4}

    def L3DMM_loss(self, alpha_true, alpha_pred):
        return reduce_sum(square(alpha_true - alpha_pred))

    def L_lmk_loss(self, L_true, L_pred):
        return self.huber(L_true, L_pred)
    
    def L3DMM_lmk_loss(self, alpha_true, alpha_hat):
        return reduce_sum(square(alpha_true - alpha_hat))
    
    def Lg_loss(self, alpha, alpha_hat):
        return reduce_sum(square(alpha - alpha_hat))
    
