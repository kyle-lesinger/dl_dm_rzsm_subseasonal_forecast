#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle
import addPredictors as pred

def load_all_data_EX0(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both,addtl_experiment,experiment_test):
    
    channel_list = []

    if addtl_experiment == False:
        assert lead >=1, 'Lead must be >=1, do not look at Week 0'
        
        lead_list = np.arange(1,lead+1)

        print(f'Only adding Reforecast data as input for leads {list(lead_list)}.')

        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],lead)) #We are adding both RZSM and/or Tmax (so use * 2)
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead))

        #Add reforecast data
        for idx,lead in enumerate(lead_list):
            channel_list.append(f'RZSM_ref_lead{lead}')

            training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx)

        print(f'Index idx value is {idx}. Done adding RZSM obs.')    

    
    elif addtl_experiment == True:
        print(f'Only adding Reforecast data as input for lead {lead}.')

        lead_add = 1 #We are only adding the current week of reforecast
        lead_list = [lead]

        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],lead_add)) #We are adding both RZSM and Tmax (so use * 2)
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead_add))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead_add))

        channel_list.append(f'RZSM_ref_lead{lead}')

        training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx)

    return(training_input, validation_input, testing_input, channel_list)



def load_all_data_EX1_EX2_EX3_EX4(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both):

    channel_list = []

    
    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5))
    
    var_list = ['pwat','spfh','tmax','diff_temp','z200']
    
    #Observations RZSM
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} RZSM obs : {training_input[:,:,:,idx]}')
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    print(f'Index idx value is {idx}. Done adding RZSM obs.')

     #Add precipitatable water (pwat)
    for lag in observation_lag_list_not_RZSM:
        channel_list.append(f'pwat_obs_lag{lag}')
        idx+=1 
        #we are going to add each channel individually. Keep track of idx value
        #Observations pwat
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} pwat obs : {training_input[:,:,:,idx]}')
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)   
    
    print(f'Index idx value is {idx}. Done adding pwat.')
    
    #Add surface_specific humidity (spfh)
    for lag in observation_lag_list_not_RZSM:
        channel_list.append(f'spfh_obs_lag{lag}')
        idx+=1
        #Observations spfh
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} spfh obs : {training_input[:,:,:,idx]}')
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)   
    
    print(f'Index idx value is {idx}. Done adding spfh.')
    
    #Add 2m maximum temperature (tmax)
    for lag in observation_lag_list_not_RZSM:
        channel_list.append(f'tmax_obs_lag{lag}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} tmax obs : {training_input[:,:,:,idx]}')
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)  
            
    print(f'Index idx value is {idx}. Done adding Tmax.')
    #Add 2m difference in maximum and minimum temperature (diff_temp). Which was Tmax - Tmin
    for lag in observation_lag_list_not_RZSM:
        channel_list.append(f'diff_temp_obs_lag{lag}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} diff temp obs : {training_input[:,:,:,idx]}')
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)  
    
    print(f'Index idx value is {idx}. Done adding diff_temp.')
    #Add geopoential height (z200)
    for lag in observation_lag_list_not_RZSM:
        channel_list.append(f'z200_obs_lag{lag}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            #print(f'\n\n Idx {idx} z200 obs : {training_input[:,:,:,idx]}')
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)  
    
    print(f'Index idx value is {idx}.Done adding z200.')
            
    #training_input = tf.convert_to_tensor(training_input,dtype=tf.float32)
    #validation_input = tf.convert_to_tensor(validation_input,dtype=tf.float32)
    #testing_input = tf.convert_to_tensor(testing_input,dtype=tf.float32)

    return(training_input, validation_input, testing_input,channel_list)



def load_all_data_EX5_EX6_EX7_EX8(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both):
    print('\nUsing observations as predictions and forecast lead week 1\n')
    #Observations RZSM
    '''We need to combine all the RZSM files into 1 single dictionary for later processing'''
    channel_list = []
    var_list = ['pwat','spfh','tmax','diff_temp','z200']
    
        
    if lead ==0:
        if RZSM_or_Tmax_or_both == 'RZSM':
            lead_add = 1
        elif RZSM_or_Tmax_or_both == 'both':
            lead_add = 2
    else:
        if RZSM_or_Tmax_or_both == 'RZSM':
            lead_add = 1
    
    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+ lead_add))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list) + lead_add))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+ lead_add))
    
    print(f'Training shape: {training_input.shape}')
    
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
    
    print(f'Index idx value is {idx}. Done adding RZSM observation lags.')
    
    if RZSM_or_Tmax_or_both == 'both':
        #Add 2m maximum temperature (tmax)
        for i in observation_lag_list_not_RZSM:
            channel_list.append(f'tmax_obs_lag{i}')
            idx+=1
            #Observations tmax
            with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
                training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
            with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
                validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
            with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
                testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)  
    
        print(f'Index idx value is {idx}. Done adding Tmax observation lags.')
    
    #Reforecast RZSM
    lead_list = [lead]
    
    for i in lead_list:
        channel_list.append(f'RZSM_ref_lead{i}')
        idx+=1
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
    
    print(f'Index idx value is {idx}. Done adding RZSM reforecast leads.')
    
    if RZSM_or_Tmax_or_both == 'both':
        #Reforecast tmax
        for i in (lead_list):
            channel_list.append(f'tmax_ref_lead{i}')
            idx+=1
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_training.pickle', 'rb') as handle:
                training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_validation.pickle', 'rb') as handle:
                validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_testing.pickle', 'rb') as handle:
                testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
            
        print(f'Index idx value is {idx}. Done adding Tmax reforecast leads.')
       
    #training_input = tf.convert_to_tensor(training_input,dtype=tf.float32)
    #validation_input = tf.convert_to_tensor(validation_input,dtype=tf.float32)
    #testing_input = tf.convert_to_tensor(testing_input,dtype=tf.float32)

    return(training_input, validation_input, testing_input, channel_list)


def load_all_data_EX9_EX10_EX11_EX12(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both):
    '''We need to combine all the RZSM files and all the observation data (pwat, spfh, tmax, diff_temp, z200) into one file. Also adding the RZSM reforecast data from RZSM and Tmax '''

    channel_list = []
    var_list = ['pwat','spfh','tmax','diff_temp','z200']
    
    if RZSM_or_Tmax_or_both == 'RZSM':
        lead_add = 1
    elif RZSM_or_Tmax_or_both == 'both':
        lead_add = 2
    lead_list = [0]

    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+(lead_add)))
    print(f'Training input shape = {training_input.shape}')
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+(lead_add)))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+(lead_add)))
    
    #Add RZSM observations first
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
            
        with open(f'{input_directory}/OBS_RZSM_GLEAM_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    print(f'Index idx value is {idx}. Done adding RZSM obs.')
    
    
    #Add precipitatable water (pwat)
    for i in observation_lag_list_not_RZSM:
        channel_list.append(f'pwat_obs_lag{i}')
        idx+=1 #we are going to add each channel individually. Keep track of idx value
        #Observations pwat
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_pwat_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
    
    print(f'Index idx value is {idx}. Done adding pwat.')
    
    #Add surface_specific humidity (spfh)
    for i in observation_lag_list_not_RZSM:
        channel_list.append(f'spfh_obs_lag{i}')
        idx+=1
        #Observations spfh
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_spfh_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)   
    
    print(f'Index idx value is {idx}. Done adding spfh.')
    
    #Add 2m maximum temperature (tmax)
    for i in observation_lag_list_not_RZSM:
        channel_list.append(f'tmax_obs_lag{i}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_tmax_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)  
            
    print(f'Index idx value is {idx}. Done adding Tmax.')
    #Add 2m difference in maximum and minimum temperature (diff_temp). Which was Tmax - Tmin
    for i in observation_lag_list_not_RZSM:
        channel_list.append(f'diff_temp_obs_lag{i}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_diff_temp_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)  
    
    print(f'Index idx value is {idx}. Done adding diff_temp.')
    #Add geopoential height (z200)
    for i in observation_lag_list_not_RZSM:
        channel_list.append(f'z200_obs_lag{i}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)
        with open(f'{input_directory}/OBS_geopotential_z200_ERA5_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{i}'].astype(np.float32)  
    
    print(f'Index idx value is {idx}.Done adding z200.')
    
    #Add RZSM reforecast week 1 through N
        
    for i in lead_list:
        channel_list.append(f'RZSM_ref_lead{i}')
        idx+=1
        #Observations tmax
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_training.pickle', 'rb') as handle:
            training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_validation.pickle', 'rb') as handle:
            validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
        with open(f'{input_directory}/REFORECAST_RZSM_GEFSv12_grid_cell_standardization_testing.pickle', 'rb') as handle:
            testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
    
    print(f'Index idx value is {idx}. Done adding RZSM reforecast.')
    #Add Tmax reforecast week 1
    if RZSM_or_Tmax_or_both == 'both':
        for i in lead_list:
            channel_list.append(f'tmax_ref_lead{i}')
            idx+=1
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_training.pickle', 'rb') as handle:
                training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_validation.pickle', 'rb') as handle:
                validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)
            with open(f'{input_directory}/REFORECAST_tmax_GEFSv12_grid_cell_standardization_testing.pickle', 'rb') as handle:
                testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{i}'].astype(np.float32)

        print(f'Index idx value is {idx}. Done adding Tmax reforecast.')
    
    #training_input = tf.convert_to_tensor(training_input,dtype=tf.float32)
    #validation_input = tf.convert_to_tensor(validation_input,dtype=tf.float32)
    #testing_input = tf.convert_to_tensor(testing_input,dtype=tf.float32)

    return(training_input, validation_input, testing_input,channel_list)

