import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from custom_losses import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights = None,input_size = (None,None,3), nr_of_stacks = 1):

    ## Leaky Relu allows a small gradient when the unit is not active:
    ## f(x) = alpha * x for x < 0, f(x) = x for x >= 0.

    # alpha: Slope in the negative part of the input space.
    # Usually a small positive value.
    # Setting alpha to 0.0 corresponds to a ReLU,
    # setting alpha to 1.0 corresponds to the identity function.
    alpha = 0.1

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv1)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv2)
    conv2 = LeakyReLU(alpha=alpha)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv3)
    conv3 = LeakyReLU(alpha=alpha)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv4)
    conv4 = LeakyReLU(alpha=alpha)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU(alpha=alpha)(conv5)
    conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU(alpha=alpha)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = LeakyReLU(alpha=alpha)(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU(alpha=alpha)(conv6)

    up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = LeakyReLU(alpha=alpha)(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU(alpha=alpha)(conv7)

    up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = LeakyReLU(alpha=alpha)(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU(alpha=alpha)(conv8)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU(alpha=alpha)(conv8)

    up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = LeakyReLU(alpha=alpha)(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU(alpha=alpha)(conv9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=alpha)(conv9)
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU(alpha=alpha)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    combined_output = conv10
    next_input=conv10

    #if(nr_of_stacks>1):
    for i in range(nr_of_stacks-1):
        inputs_2 = next_input
        conv1_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs_2)
        conv1_2 = LeakyReLU(alpha=alpha)(conv1_2)
        conv1_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv1_2)
        conv1_2 = LeakyReLU(alpha=alpha)(conv1_2)
        pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
        conv2_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1_2)
        conv2_2 = LeakyReLU(alpha=alpha)(conv2_2)
        conv2_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv2_2)
        conv2_2 = LeakyReLU(alpha=alpha)(conv2_2)
        pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
        conv3_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2_2)
        conv3_2 = LeakyReLU(alpha=alpha)(conv3_2)
        conv3_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv3_2)
        conv3_2 = LeakyReLU(alpha=alpha)(conv3_2)
        pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
        conv4_2 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3_2)
        conv4_2 = LeakyReLU(alpha=alpha)(conv4_2)
        conv4_2 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal', dilation_rate=2)(conv4_2)
        conv4_2 = LeakyReLU(alpha=alpha)(conv4_2)
        drop4_2 = Dropout(0.5)(conv4_2)
        pool4_2 = MaxPooling2D(pool_size=(2, 2))(drop4_2)

        conv5_2 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4_2)
        conv5_2 = LeakyReLU(alpha=alpha)(conv5_2)
        conv5_2 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5_2)
        conv5_2 = LeakyReLU(alpha=alpha)(conv5_2)
        drop5_2 = Dropout(0.5)(conv5_2)

        up6_2 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5_2))
        up6_2 = LeakyReLU(alpha=alpha)(up6_2)
        merge6_2 = concatenate([drop4_2,up6_2], axis = 3)
        conv6_2 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6_2)
        conv6_2 = LeakyReLU(alpha=alpha)(conv6_2)
        conv6_2 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6_2)
        conv6_2 = LeakyReLU(alpha=alpha)(conv6_2)

        up7_2 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_2))
        up7_2 = LeakyReLU(alpha=alpha)(up7_2)
        merge7_2 = concatenate([conv3_2,up7_2], axis = 3)
        conv7_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7_2)
        conv7_2 = LeakyReLU(alpha=alpha)(conv7_2)
        conv7_2 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7_2)
        conv7_2 = LeakyReLU(alpha=alpha)(conv7_2)

        up8_2 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_2))
        up8_2 = LeakyReLU(alpha=alpha)(up8_2)
        merge8_2 = concatenate([conv2_2,up8_2], axis = 3)
        conv8_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8_2)
        conv8_2 = LeakyReLU(alpha=alpha)(conv8_2)
        conv8_2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8_2)
        conv8_2 = LeakyReLU(alpha=alpha)(conv8_2)

        up9_2 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_2))
        up9_2 = LeakyReLU(alpha=alpha)(up9_2)
        merge9_2 = concatenate([conv1_2,up9_2], axis = 3)
        conv9_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9_2)
        conv9_2 = LeakyReLU(alpha=alpha)(conv9_2)
        conv9_2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9_2)
        conv9_2 = LeakyReLU(alpha=alpha)(conv9_2)
        conv9_2 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9_2)
        conv9_2 = LeakyReLU(alpha=alpha)(conv9_2)
        conv10_2 = Conv2D(1, 1, activation = 'sigmoid')(conv9_2)
        next_input=conv10_2
        combined_output = concatenate([combined_output, next_input], axis=3)

    model = Model(input = inputs, output = combined_output)


    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', grid_loss, kaggle_metric])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


