import tensorflow as tf
from tensorflow.math import square, reduce_sum
from tensorflow.keras.losses import Loss, Huber

class Synergy_Loss(Loss):
    def __init__(self, lamda_1=0.02, lamda_2=0.03, lamda_3=0.02, lamda_4=0.001, **kwargs):
        super(Synergy_Loss, self).__init__(**kwargs)
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.lamda_3 = lamda_3
        self.lamda_4 = lamda_4
        self.huber = Huber()

    def call(self, y_true, y_pred):
        pose_3DMM_true, L_true = y_true
        pose_3DMM, Lc, Lr, pose_3DMM_hat, alpha_exp, alpha_shp = y_pred
        L3DMM = self.L3DMM_loss(pose_3DMM_true, pose_3DMM)
        L_lmk = self.L_lmk_loss(L_true, Lr)
        L3DMM_lmk = self.L3DMM_lmk_loss(pose_3DMM_true, pose_3DMM_hat)
        Lg = self.Lg_loss(pose_3DMM, pose_3DMM_hat)
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
        return self.huber(L_true - L_pred)
    
    def L3DMM_lmk_loss(self, alpha_true, alpha_hat):
        return reduce_sum(square(alpha_true - alpha_hat))
    
    def Lg_loss(self, alpha, alpha_hat):
        return reduce_sum(square(alpha - alpha_hat))
    
