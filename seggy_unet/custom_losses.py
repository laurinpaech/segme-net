import numpy as np
from keras.layers import *
from keras.optimizers import *


def grid_loss(y_true, y_pred):
    # differentiable
    myfilter=tf.ones([16,16,1,1])
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_pred_reduced=tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    return tf.losses.mean_squared_error(y_true_reduced,y_pred_reduced)


def kaggle_metric(y_true, y_pred):
    # not differentiable
    myfilter=tf.ones([16,16,1,1])
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_true_reduced = np.where(y_true_reduced > 3, 1, 0)
    y_pred_reduced=tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_pred_reduced = np.where(y_pred_reduced > 3, 1, 0)
    return tf.losses.mean_squared_error(y_true_reduced,y_pred_reduced)

