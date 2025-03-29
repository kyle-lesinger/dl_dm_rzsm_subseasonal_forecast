#!/usr/bin/env python3

import pickle
import numpy as np


def add_obs_RZSM_by_lag(training_input, validation_input, testing_input,lag, input_directory, idx):
    with open(f'{input_directory}/obs_soilw_bgrnd_GLEAM_mean_training.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    with open(f'{input_directory}/obs_soilw_bgrnd_GLEAM_mean_validation.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    with open(f'{input_directory}/obs_soilw_bgrnd_GLEAM_mean_testing.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    return(training_input, validation_input, testing_input)


def return_masking_objects_for_RZSM(input_directory):
    #For masking
    with open(f'{input_directory}/obs_soilw_bgrnd_GLEAM_mean_training.pickle', 'rb') as handle:
        RZSM_train_obs = np.expand_dims(pickle.load(handle)[f'Lag-1'].astype(np.float32),-1)
        
    #For masking
    with open(f'{input_directory}/obs_soilw_bgrnd_GLEAM_mean_validation.pickle', 'rb') as handle:
        RZSM_mean_validation_obs = np.expand_dims(pickle.load(handle)[f'Lag-1'].astype(np.float32),-1)

    return(RZSM_train_obs, RZSM_mean_validation_obs)


def add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx):
    print(f'Loading observations for {variable} and lag {lag}')
    with open(f'{input_directory}/obs_{variable}_ERA5_mean_training.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
    with open(f'{input_directory}/obs_{variable}_ERA5_mean_validation.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
    with open(f'{input_directory}/obs_{variable}_ERA5_mean_testing.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32)
        
    return(training_input, validation_input, testing_input)


def return_channel_name(variable):
    if variable == 'pwat_eatm':
        return('pwat')
    elif variable == 'spfh_2m':
        return('spfh')
    elif variable == 'tmax_2m':
        return('tmax')
    elif variable == 'diff_temp_2m':
        return('diff_temp')
    elif variable == 'hgt_pres':
        return('z200')
    elif variable == 'soilw_bgrnd':
        return('RZSM')


def add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory, variable, idx):
    with open(f'{input_directory}/reforecast_{variable}_GEFSv12_mean_training.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32)
        
    with open(f'{input_directory}/reforecast_{variable}_GEFSv12_mean_validation.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32)
        
    with open(f'{input_directory}/reforecast_{variable}_GEFSv12_mean_testing.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32)
        
    return(training_input, validation_input, testing_input)

