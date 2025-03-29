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

def conv_batchnorm_relu_block_SE_and_residual_connections(input_tensor, nb_filter, dropout_rate, kernel_norm, kernel_size=3,kernel_initializer=kernel_initializer):
    depth_multiplier=1
    
    shortcut = Conv2D(nb_filter, kernel_size=(1, 1), padding='same', strides=1, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)
    
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    #x = DepthwiseConv2D(3, padding ='same', depth_multiplier=depth_multiplier, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(x,training=True)
    #x = Dropout(dropout_rate)(x, training =True) #add dropout 

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(x,training=True)
    
    residual_SE_block = Add()([shortcut, x])
    residual_SE_block = keras_cv.layers.SqueezeAndExcite2D(residual_SE_block.shape[-1])(residual_SE_block)
    
    return residual_SE_block

def inception_block(prevlayer, a, b,dropout_rate, kernel_norm, depth_multiplier = False):
    shortcut = Conv2D(a,kernel_size=(1, 1), padding='same', strides=1, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)
    
    if depth_multiplier == True:
        depth_multiplier=16
    else:
        depth_multiplier=2
    
    conva = DepthwiseConv2D(3, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    conva = BatchNormalization()(conva)
    conva = tf.keras.activations.relu(conva)
    conva = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(conva,training=True)
    #conva = Dropout(dropout_rate)(conva, training =True) #add dropout
    
    conva = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conva)
    conva = BatchNormalization()(conva)
    conva = tf.keras.activations.relu(conva)
    conva = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(conva,training=True)
    #conva = Dropout(dropout_rate)(conva, training =True) #add dropout

    convb = DepthwiseConv2D(5, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convb = BatchNormalization()(convb)
    convb = tf.keras.activations.relu(convb)
    convb = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convb,training=True)
    #convb = Dropout(dropout_rate)(convb, training =True) #add dropout
    
    convb = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convb)
    convb = BatchNormalization()(convb)
    convb = tf.keras.activations.relu(convb)
    convb = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convb,training=True)
    #convb = Dropout(dropout_rate)(convb, training =True) #add dropout

    convc = DepthwiseConv2D(7, padding='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convc = BatchNormalization()(convc)
    convc = tf.keras.activations.relu(convc)
    convc = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convc,training=True)
    #convc = Dropout(dropout_rate)(convc, training =True) #add dropout
    
    convc = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convc)
    convc = BatchNormalization()(convc)
    convc = tf.keras.activations.relu(convc)
    convc = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convc,training=True)
    #convc = Dropout(dropout_rate)(convc, training =True) #add dropout

    # if True == pooling:
    #     convd = MaxPooling2D(pool_size=(2, 2))(convd)
    
    #Max pool
    conve = MaxPool2D((5,5), strides=(1, 1), padding='same')(prevlayer)
    conve = Conv2D(a,(1, 1), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conve)
    conve = BatchNormalization()(conve)
    conve = tf.keras.activations.relu(conve)
    conve = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(conve,training=True)
    #onvd = Dropout(dropout_rate)(convd, training =True) #add dropout
    
    conve = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conve)
    conve = BatchNormalization()(conve)
    conve = tf.keras.activations.relu(conve)
    conve = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(conve,training=True)
    #onvd = Dropout(dropout_rate)(convd, training =True) #add dropout

    up = concatenate([conva, convb, convc, conve])
    
    residual_block = Concatenate()([shortcut, up])
    #Test with a squeeze and excitation block
    residual_block = keras_cv.layers.SqueezeAndExcite2D(residual_block.shape[-1])(residual_block)
    
    return residual_block

def model_build_func(inputs,kernel_norm, output_channels,var_name, number_of_UNET_backbone_max_pool,using_deep_supervision=True,kernel_initializer=kernel_initializer ):

    dropout_rate_initial = 0.1
    dropout_rate_later = 0.25
    #nb_filter = [16,32,64,128]
    nb_filter = [32,32,64,128]

    # Set image data format to channels first
    global bn_axis
    
    k.backend.set_image_data_format("channels_last")
    bn_axis = -1

    conv1_1 = inception_block(inputs, nb_filter[0], nb_filter[0],dropout_rate=dropout_rate_initial,depth_multiplier=True, kernel_norm =kernel_norm)
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)
    
    conv2_1 = inception_block(pool1, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv1_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    # conv1_2=inception_block(conv1_2, nb_filter[0], nb_filter[0],dropout_rate=dropout_rate_initial,depth_multiplier=False)
    # conv1_2 = inception_blockz(conv1_2, nb_filter = nb_filter[1], dropout_rate = dropout_rate_later)

    conv3_1=inception_block(pool2, nb_filter[0], nb_filter[0],dropout_rate=dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    pool3 = MaxPool2D((2, 2), strides=(2, 2),)(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv2_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    # conv2_2 = inception_block(conv2_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block_SE_and_residual_connections(conv1_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    # conv1_3 = inception_block(conv1_3, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False)

    conv4_1 = inception_block(pool3, nb_filter[3], nb_filter[3], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    pool4 = MaxPool2D((2, 2), strides=(2, 2),)(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv3_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    # conv3_2 = inception_block(conv3_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2],axis=bn_axis)
    conv2_3 = inception_block(conv2_3, nb_filter[3], nb_filter[3], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=bn_axis)
    conv1_4 = inception_block(conv1_4, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)

    #conv5_1 = inception_block(pool4, nb_filter[3], nb_filter[3], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)

    #up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv5_1)
    #conv4_2 = concatenate([up4_2, conv4_1], axis=bn_axis)
    #conv4_2 = inception_block(conv4_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    # conv4_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv4_2, nb_filter=nb_filter[3], dropout_rate = dropout_rate_later)

    #up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv4_2)
    #conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=bn_axis)
    #conv3_3 = inception_block(conv3_3, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    # conv3_3 = conv_batchnorm_relu_block_SE_and_residual_connections(conv3_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later)

    #up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2),  padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv3_3)
    #conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3],  axis=bn_axis)
    #conv2_4 = inception_block(conv2_4, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    # conv2_4 =  conv_batchnorm_relu_block_SE_and_residual_connections(conv2_4, nb_filter=nb_filter[1], dropout_rate = dropout_rate_later)

    #up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_4)
    #conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=bn_axis)
    # conv1_5 = inception_block(conv1_5, nb_filter[0], nb_filter[0], dropout_rate = dropout_rate_later,depth_multiplier=False)
    #conv1_5 =  conv_batchnorm_relu_block_SE_and_residual_connections(conv1_5, nb_filter=nb_filter[0], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    nestnet_output_1 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_1',padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_2)
    nestnet_output_2 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_2', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32 )(conv1_3)
    nestnet_output_3 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_3', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)
    #nestnet_output_4 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_4', padding='same',kernel_initializer=kernel_initializer)(conv1_5)
    

    if number_of_UNET_backbone_max_pool == 4:
        return(nestnet_output_1,nestnet_output_2,nestnet_output_3)
    
        
        
                # model = Model(inputs=inputs, outputs=[nestnet_output_1,
                #                                     nestnet_output_2,
                #                                     nestnet_output_3,
                #                                     nestnet_output_4])
        # else:
        #     model = Model(inputs=inputs, outputs=nestnet_output_4)

        # return model
