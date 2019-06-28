import numpy as np
from keras.layers import *
from keras.optimizers import *


def grid_loss(y_true, y_pred):
    # differentiable
    myfilter=tf.ones([16,16,tf.shape(y_true)[3],1])
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_pred_reduced=tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    return tf.losses.mean_squared_error(y_true_reduced,y_pred_reduced)


def kaggle_metric(y_true, y_pred):
    # not differentiable
    myfilter=tf.ones([16,16,tf.shape(y_true)[3],1])
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_true_reduced = tf.where(y_true_reduced > 3, tf.ones(tf.shape(y_true_reduced)), tf.zeros(tf.shape(y_true_reduced)))
    y_pred_reduced=tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_pred_reduced = tf.where(y_pred_reduced > 3, tf.ones(tf.shape(y_true_reduced)), tf.zeros(tf.shape(y_true_reduced)))

    # only use results from last unet
    return tf.losses.mean_squared_error(y_true_reduced[:,:,:,tf.shape(y_true_reduced)[3]-1],y_pred_reduced[:,:,:,tf.shape(y_true_reduced)[3]-1])

