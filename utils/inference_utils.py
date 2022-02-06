import numpy as np
from model.morhaple_face_model import PCA
import tensorflow as tf

def predict_lmks(param_3DMM, roi_box):
    lmks = tf.squeeze(PCA()(param_3DMM), 0).numpy()
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 128
    scale_y = (ey - sy) / 128
    lmks[:, 0] = lmks[:, 0] * scale_x + sx
    lmks[:, 1] = lmks[:, 1] * scale_y + sy
    s = (scale_x + scale_y) / 2
    lmks[:, 2] *= s
    return lmks