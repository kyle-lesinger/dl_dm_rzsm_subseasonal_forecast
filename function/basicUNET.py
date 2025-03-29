#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle
import keras as k
from keras.layers import Conv2D, Input, AvgPool2D, MaxPool2D, Concatenate, Add, Dropout, BatchNormalization, Conv2DTranspose, Activation, DepthwiseConv2D, concatenate
import keras_cv

kernel_initializer = 'glorot_uniform'

def conv_batchnorm_relu_block(input_tensor, nb_filter, dropout_rate, kernel_norm, kernel_size=3,kernel_initializer=kernel_initializer):

    
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x, training =True) #add dropout 

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x, training =True) #add dropout 

    return x


def model_build_func(inputs,kernel_norm, output_channels,var_name, number_of_UNET_backbone_max_pool,using_deep_supervision=True,kernel_initializer=kernel_initializer ):

    dropout_rate_initial = 0.1
    dropout_rate_later = 0.25
    #nb_filter = [16,32,64,128]
    nb_filter = [32,64,128,256]

    # Set image data format to channels first
    global bn_axis
    
    k.backend.set_image_data_format("channels_last")
    bn_axis = -1

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)
    
    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)

    conv3_1=conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    pool3 = MaxPool2D((2, 2), strides=(2, 2),)(conv3_1)


    # conv1_3 = inception_block(conv1_3, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False)

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    # conv3_2 = inception_block(conv3_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1],axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1], axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(up1_4, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    nestnet_output_1 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_1', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)
    nestnet_output_2 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_2', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)
    nestnet_output_3 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_3', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)

    if number_of_UNET_backbone_max_pool == 4:
        return(nestnet_output_1,nestnet_output_2,nestnet_output_3)
    
        
        
                # model = Model(inputs=inputs, outputs=[nestnet_output_1,
                #                                     nestnet_output_2,
                #                                     nestnet_output_3,
                #                                     nestnet_output_4])
        # else:
        #     model = Model(inputs=inputs, outputs=nestnet_output_4)

        # return model
