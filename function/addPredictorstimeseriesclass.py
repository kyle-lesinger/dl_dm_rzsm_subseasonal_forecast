#!/usr/bin/env python3

import pickle
import numpy as np


def add_obs_RZSM_by_lag(training_input, validation_input, testing_input,lag, input_directory, idx, final_testing_year):

    if (final_testing_year == None) or (final_testing_year == 2019):
        add_year = ''
    else:
        add_year = f'_{final_testing_year}'
        
    with open(f'{input_directory}/obs_RZSM_standardized_GLEAM_training{add_year}.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    with open(f'{input_directory}/obs_RZSM_standardized_GLEAM_validation{add_year}.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    with open(f'{input_directory}/obs_RZSM_standardized_GLEAM_testing{add_year}.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    return(training_input, validation_input, testing_input)


def return_masking_objects_for_RZSM(input_directory,final_testing_year):
    
    if (final_testing_year == None) or (final_testing_year == 2019):
        add_year = ''
    else:
        add_year = f'_{final_testing_year}'
    #For masking
    with open(f'{input_directory}/obs_RZSM_standardized_GLEAM_training{add_year}.pickle', 'rb') as handle:
        RZSM_train_obs = np.expand_dims(pickle.load(handle)[f'Lag-1'].astype(np.float32),-1)
        
    #For masking
    with open(f'{input_directory}/obs_RZSM_standardized_GLEAM_validation{add_year}.pickle', 'rb') as handle:
        RZSM_validation_obs = np.expand_dims(pickle.load(handle)[f'Lag-1'].astype(np.float32),-1)

    return(RZSM_train_obs, RZSM_validation_obs)


def add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx,final_testing_year):
    
    if (final_testing_year == None) or (final_testing_year == 2019):
        add_year = ''
    else:
        add_year = f'_{final_testing_year}'
        
    print(f'Loading observations for {variable} and lag {lag}')
    with open(f'{input_directory}/obs_{variable}_ERA5_training{add_year}.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
    with open(f'{input_directory}/obs_{variable}_ERA5_validation{add_year}.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
    with open(f'{input_directory}/obs_{variable}_ERA5_testing{add_year}.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lag{lag}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    return(training_input, validation_input, testing_input)


def return_channel_name(variable):
    return(variable)



def add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory, variable, idx, ref_source,final_testing_year):
    
    if (final_testing_year == None) or (final_testing_year == 2019):
        add_year = ''
    else:
        add_year = f'_{final_testing_year}'
        
    with open(f'{input_directory}/reforecast_{variable}_{ref_source}_training{add_year}.pickle', 'rb') as handle:
        training_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    with open(f'{input_directory}/reforecast_{variable}_{ref_source}_validation{add_year}.pickle', 'rb') as handle:
        validation_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    with open(f'{input_directory}/reforecast_{variable}_{ref_source}_testing{add_year}.pickle', 'rb') as handle:
        testing_input[:,:,:,idx] = pickle.load(handle)[f'Lead{lead}'].astype(np.float32).unstack('combine_models').mean(dim='M').transpose('S','Y','X')
        
    return(training_input, validation_input, testing_input)

