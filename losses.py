from model.morhaple_face_model import PCA
from data_generator.labels_preprocessing import label_to_3DMM, label_to_pt2d
import tensorflow as tf
from tensorflow.math import square, reduce_sum, log, abs, less, greater_equal
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError

class Synergy_Loss(Loss):
    def __init__(self, lamda_1=0.02, lamda_2=0.05, lamda_3=0.02,\
         lamda_4=0.001, input_shape=(224, 224, 3),\
              omega=10, epsilon=2, **kwargs):
        super(Synergy_Loss, self).__init__(**kwargs)
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.lamda_3 = lamda_3
        self.lamda_4 = lamda_4
        self.huber = Huber()
        self.mse = MeanSquaredError()
        self.input_shape = input_shape
        self.omega = omega
        self.epsilon = epsilon
        self.log_term = log(1 + self.omega/self.epsilon)
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

        L3DMM = self.L3DMM_loss(y_gt, y_pred)
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

    def L3DMM_loss(self, y_gt, y_pred):
        return self.mse(y_gt[:, 0:12], y_pred[:, 0:12]) #+ self.mse(y_gt[:, 12:], y_pred[:, 12:])

    def L_lmk_loss(self, L_gt, L_pred):
        y_pred = tf.reshape(L_pred, (-1, L_pred.shape[1]*3))
        y_gt = tf.reshape(L_gt, (-1, L_gt.shape[1]*3))
        delta_y = abs(y_gt - y_pred)
        is_less_than = less(delta_y, self.omega)
        loss1 = self.omega * log(1 + delta_y/self.epsilon)
        C = self.omega * (1 - self.log_term)
        loss2 = delta_y - C
        loss = tf.where(is_less_than, loss1, loss2)
        return loss
    
    def L3DMM_lmk_loss(self, alpha_true, alpha_hat):
        return reduce_sum(square(alpha_true - alpha_hat))
    
    def Lg_loss(self, alpha, alpha_hat):
        return reduce_sum(square(alpha - alpha_hat))
    
