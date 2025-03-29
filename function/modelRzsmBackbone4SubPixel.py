#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import keras as k
from keras.layers import Conv2D, Input, AvgPool2D, MaxPool2D, Concatenate, Add, Dropout, BatchNormalization, Conv2DTranspose, Activation, DepthwiseConv2D, concatenate
import keras_cv
from keras.models import Model

kernel_initializer = 'glorot_uniform'


def inception_block(prevlayer, a, out_filter_divisible_by_4, dropout_rate, kernel_norm, depth_multiplier = False):
    shortcut = prevlayer

    if depth_multiplier == True:
        depth_multiplier=4
    else:
        depth_multiplier=2
    
    conva = DepthwiseConv2D(3, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    conva = BatchNormalization()(conva)
    conva = tf.keras.activations.sigmoid(conva)
    conva = Dropout(dropout_rate)(conva, training =True) #add dropout
    
    conva = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conva)
    #conva = BatchNormalization()(conva)
    #conva = tf.keras.activations.sigmoid(conva)
    conva = Dropout(dropout_rate)(conva, training =True) #add dropout

    convb = DepthwiseConv2D(5, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convb = BatchNormalization()(convb)
    convb = tf.keras.activations.sigmoid(convb)
    convb = Dropout(dropout_rate)(convb, training =True) #add dropout
    
    convb = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convb)
    #convb = BatchNormalization()(convb)
    #convb = tf.keras.activations.sigmoid(convb)
    convb = Dropout(dropout_rate)(convb, training =True) #add dropout

    convc = DepthwiseConv2D(7, padding='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convc = BatchNormalization()(convc)
    convc = tf.keras.activations.sigmoid(convc)
    convc = Dropout(dropout_rate)(convc, training =True) #add dropout
    
    convc = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convc)
    #convc = BatchNormalization()(convc)
    #convc = tf.keras.activations.sigmoid(convc)
    convc = Dropout(dropout_rate)(convc, training =True) #add dropout

    # if True == pooling:
    #     convd = MaxPooling2D(pool_size=(2, 2))(convd)
    
    #Max pool
    convd = MaxPool2D((5,5), strides=(1, 1), padding='same')(prevlayer)
    convd = Conv2D(a,(1, 1), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convd)
    convd = BatchNormalization()(convd)
    convd = tf.keras.activations.sigmoid(convd)
    convd = Dropout(dropout_rate)(convd, training =True) #add dropout
    
    convd = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convd)
    #convd = BatchNormalization()(convd)
    #convd = tf.keras.activations.sigmoid(convd)
    convd = Dropout(dropout_rate)(convd, training =True) #add dropout

    up = concatenate([conva, convb, convc, convd])
    
    residual_block = Concatenate()([shortcut, up]) #Can't add, the shapes aren't the same. Can only concatenate
    residual_block = BatchNormalization()(residual_block) #Only batchnorm after all the inputs are concatenated
    residual_block = tf.keras.activations.sigmoid(residual_block)
    
    residual_SE_block = keras_cv.layers.SqueezeAndExcite2D(filters=residual_block.shape[-1],squeeze_activation="sigmoid")(residual_block)

    #return a filter list divisible by 4 for some operations
    if out_filter_divisible_by_4 !=0: 
        final_out = Conv2D(out_filter_divisible_by_4,(3, 3), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(residual_SE_block)
        final_out = BatchNormalization()(final_out)
        final_out = tf.keras.activations.sigmoid(final_out)
        return (final_out)
    else:
        return residual_SE_block







def model_build_func(inputs,kernel_norm, output_channels,var_name, number_of_UNET_backbone_max_pool,using_deep_supervision=True,kernel_initializer=kernel_initializer ):

    # Set image data format to channels first
    global bn_axis
    
    divide_channels = 1 #divide_channels == 2 to shrink to 10,000,000 parameters. If divide_channels ==1, there is 25,000,000 parameters
    
    k.backend.set_image_data_format("channels_last")
    bn_axis = -1
    
    conv1_1 = inception_block(prevlayer = inputs, a=nb_filter[0], out_filter_divisible_by_4 = 64//divide_channels, dropout_rate=dropout_rate_initial,depth_multiplier=True, kernel_norm =kernel_norm)
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)
    
    conv2_1 = inception_block(pool1, nb_filter[1], out_filter_divisible_by_4 = 256//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)
    
    up1_2 = tf.nn.depth_to_space(conv2_1, block_size=2)
    conv1_2 = concatenate([up1_2, conv1_1], axis=bn_axis)
    #conv1_2 = inception_block(conv1_2,a =  nb_filter[0], out_filter_divisible_by_4 = 64//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    conv3_1=inception_block(pool2, a= nb_filter[2], out_filter_divisible_by_4 = 1024//divide_channels,dropout_rate=dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    pool3 = MaxPool2D((2, 2), strides=(2, 2),)(conv3_1)
    
    up2_2 = tf.nn.depth_to_space(conv3_1, block_size=2)
    conv2_2 = concatenate([up2_2, conv2_1], axis=bn_axis)
    conv2_2 = inception_block(conv2_2, a = nb_filter[1], out_filter_divisible_by_4 = 256//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    up1_3 = tf.nn.depth_to_space(conv2_2, block_size=2) 
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=bn_axis)
    #conv1_3 = inception_block(conv1_3, a = nb_filter[1], out_filter_divisible_by_4 = 64//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    conv4_1 = inception_block(pool3, a= nb_filter[3], out_filter_divisible_by_4 = 4096//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    up3_2 = tf.nn.depth_to_space(conv4_1, block_size=2)
    conv3_2 = concatenate([up3_2, conv3_1], axis=bn_axis)
    conv3_2 = inception_block(conv3_2, a= nb_filter[2], out_filter_divisible_by_4 = 1024//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm=kernel_norm)
    
    up2_3 = tf.nn.depth_to_space(conv3_2, block_size=2) 
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2],axis=bn_axis)
    conv2_3 = inception_block(conv2_3, a = nb_filter[1], out_filter_divisible_by_4 = 256//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    up1_4 = tf.nn.depth_to_space(conv2_3, block_size=2)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=bn_axis)
    #conv1_4 = inception_block(conv1_4, a =nb_filter[0], out_filter_divisible_by_4 = 64//divide_channels, dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    
    
    nestnet_output_1 = Conv2D(output_channels, (1, 1), activation='sigmoid', name=f'RZSM_output_1', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_2)
    nestnet_output_2 = Conv2D(output_channels, (1, 1), activation='sigmoid', name=f'RZSM_output_2', padding='same',kernel_initializer=kernel_initializer ,dtype=tf.float32)(conv1_3)
    nestnet_output_3 = Conv2D(output_channels, (1, 1), activation='sigmoid', name=f'RZSM_output_3', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)


    if using_deep_supervision:
        if number_of_UNET_backbone_max_pool == 4:
            return(nestnet_output_1,nestnet_output_2,nestnet_output_3)
        
        
        
                #model = Model(inputs=inputs, outputs=[nestnet_output_1,
                #                                   nestnet_output_2,
                 #                                   nestnet_output_3])
        # else:
        #     model = Model(inputs=inputs, outputs=nestnet_output_4)

        # return model
