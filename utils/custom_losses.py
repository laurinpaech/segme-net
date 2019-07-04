from keras.layers import *
from keras.optimizers import *


def grid_loss(y_true, y_pred):

    # differentiable
    myfilter = tf.ones([16, 16, tf.shape(y_true)[3], 1])
    y_true_reduced = tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    y_pred_reduced = tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    return tf.losses.mean_squared_error(y_true_reduced, y_pred_reduced)


def kaggle_metric(y_true, y_pred):
    # not differentiable
    myfilter = tf.ones([16, 16, tf.shape(y_true)[3], 1])
    y_pred_filtered = tf.where(y_pred > 0.35, tf.ones(tf.shape(y_pred)), tf.zeros(tf.shape(y_pred)))

    y_pred_reduced = tf.nn.conv2d(input=y_pred_filtered, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    y_pred_reduced = tf.where(y_pred_reduced > 63, tf.ones(tf.shape(y_pred_reduced)),
                              tf.zeros(tf.shape(y_pred_reduced)))
    y_true_reduced = tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    y_true_reduced = tf.where(y_true_reduced > 63, tf.ones(tf.shape(y_true_reduced)),
                              tf.zeros(tf.shape(y_true_reduced)))

    # only use results from last unet
    return tf.losses.mean_squared_error(y_true_reduced, y_pred_reduced)


def prob_kaggle_metric(y_true, y_pred, beta=3):
    # not differentiable
    myfilter = tf.ones([16, 16, tf.shape(y_true)[3], 1])
    y_true_reduced = tf.nn.conv2d(input=y_true, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    print(np.max(y_true_reduced).eval())
    y_true_reduced = tf.where(y_true_reduced > 63, tf.ones(tf.shape(y_true_reduced)),
                              tf.zeros(tf.shape(y_true_reduced)))
    y_pred_reduced = tf.nn.conv2d(input=y_pred, filter=myfilter, padding="VALID", strides=[1, 16, 16, 1])
    print(np.max(y_pred_reduced).eval())
    y_pred_reduced = tf.where(y_pred_reduced > beta, tf.ones(tf.shape(y_pred_reduced)),
                              tf.zeros(tf.shape(y_pred_reduced)))

    # only use results from last unet
    return tf.losses.mean_squared_error(y_true_reduced, y_pred_reduced)
