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



def real_kaggle_metric(y_true, y_pred, alpha=0.5):
    # not differentiable
    # note: not really kaggle capable, only used for alpha_testing.py
    myfilter=tf.ones([16,16,1,1] ,dtype=tf.float32)
    y_pred_filtered = tf.where(y_pred > alpha, tf.ones(tf.shape(y_pred)), tf.zeros(tf.shape(y_pred)))

    y_pred_reduced=tf.nn.conv2d(input=y_pred_filtered, filter=myfilter, padding="VALID", strides=[16])
    y_pred_reduced = tf.where(y_pred_reduced > 3, tf.ones(tf.shape(y_pred_reduced)), tf.zeros(tf.shape(y_pred_reduced)))
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[16])
    y_true_reduced = tf.where(y_true_reduced > 3, tf.ones(tf.shape(y_true_reduced)), tf.zeros(tf.shape(y_true_reduced)))

    # only use results from last unet
    return tf.losses.mean_squared_error(y_true_reduced,y_pred_reduced)

def real_kaggle_metric_035(y_true, y_pred):
    # not differentiable
    myfilter=tf.ones([16,16,tf.shape(y_true)[3],1])
    y_pred_filtered = tf.where(y_pred > 0.35, tf.ones(tf.shape(y_pred)), tf.zeros(tf.shape(y_pred)))

    y_pred_reduced=tf.nn.conv2d(input=y_pred_filtered, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_pred_reduced = tf.where(y_pred_reduced > 3, tf.ones(tf.shape(y_pred_reduced)), tf.zeros(tf.shape(y_pred_reduced)))
    y_true_reduced=tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1,16,16,1])
    y_true_reduced = tf.where(y_true_reduced > 3, tf.ones(tf.shape(y_true_reduced)), tf.zeros(tf.shape(y_true_reduced)))

    # only use results from last unet
    return tf.losses.mean_squared_error(y_true_reduced,y_pred_reduced)

