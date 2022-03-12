import tensorflow as tf
from tensorflow import reduce_sum, abs, sqrt, divide
from tensorflow.keras.losses import Loss, MeanSquaredError, Reduction, Huber

class ParameterLoss(Loss):
    def __init__(self, reduction=Reduction.NONE, name='Parameter_Loss', mode='3dmm', loss='MSE'):
        super(ParameterLoss, self).__init__(name=name)
        self.mse = MeanSquaredError(reduction=reduction)
        self.huber = Huber(reduction=reduction)
        self.mode = mode
        if loss == 'MSE':
            self.loss = self.mse
        elif loss == 'Huber':
            self.loss = self.huber
    
    def call(self, y_true, y_pred):
        if self.mode == 'normal':
            loss = self.loss(y_true=y_true[:, :12], y_pred=y_pred[:, :12])\
                + self.loss(y_true=y_true[:, 12:], y_pred=y_pred[:, 12:])
        elif self.mode == '3dmm':
            loss = self.loss(y_true=y_true[:, 12:], y_pred=y_pred[:, 12:])
        return sqrt(loss)
        
    def get_config(self):
        base_config = super(ParameterLoss, self).get_config()
        return {**base_config, 'MSE': self.mse}


class WingLoss(Loss):
    def __init__(self, omega=10, epsilon=2, name='Wing_Loss'):
        super(WingLoss, self).__init__(name=name)
        self.omega = omega
        self.epsilon = epsilon
        self.log_term = tf.math.log(1 + self.omega/self.epsilon)
    
    def call(self, y_true, y_pred):
        n_points = y_pred.shape[1]
        y_true = tf.reshape(tf.transpose(y_true, perm=[0, 2, 1]), (-1, 3*n_points))
        y_pred = tf.reshape(tf.transpose(y_pred, perm=[0, 2, 1]), (-1, 3*n_points))
        delta_y = abs(y_true - y_pred)
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * tf.math.log(1 + delta_y1/self.epsilon)
        C = self.omega - self.omega * self.log_term
        loss2 = delta_y2 - C
        sums = reduce_sum(loss1) + reduce_sum(loss2)
        lengths = tf.shape(loss1)[0] + tf.shape(loss2)[0]
        lengths = tf.cast(lengths, dtype=tf.float32)
        loss_final = divide(sums, lengths)
        return loss_final

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'omega': self.omega, 'epsilon': self.epsilon, 'log_term': self.log_term}
