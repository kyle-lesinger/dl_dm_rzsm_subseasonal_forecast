#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tifffile as tiff
import bottleneck as bn
import climpred
from climpred.options import OPTIONS

#Personal functions
from . import channelExperiment as CE
from . import verifications

training_size_shape = np.array((9185,48,96))
validation_testing_size_shape = np.array((1144,48,96))


def return_seasons():
    # Define seasons
    seasons = {
        "DJF": [12, 1, 2],  # Winter
        "MAM": [3, 4, 5],   # Spring
        "JJA": [6, 7, 8],   # Summer
        "SON": [9, 10, 11]  # Fall
    }
    return seasons

def load_ERA5_data(variable):
    #Load the pre-processed files from ERA5
    return(xr.open_dataset(f'Data/ERA5/{variable}_merged.nc4'))


def create_standard_date_format(file_dates):
    out_dates = []
    for _date in file_dates:
        year_ = pd.to_datetime(_date).year
        month_ = pd.to_datetime(_date).month
        month_ = f'{month_:02}'
        day_ = pd.to_datetime(_date).day
        day_ = f'{day_:02}'
        out_dates.append(f'{year_}-{month_}-{day_}')
    return out_dates

# Loss function (CRPS experimental)

def load_verification_observations_updated(lead,verification_directory,obs_source):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    print(f'Loading obs_soilw_bgrnd_{obs_source}_lead_{lead}_train_masked.npy')
    obs_train_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_train_masked.npy').astype(np.float32)

    ################################ VALIDATION ################################
    obs_val_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_val_masked.npy').astype(np.float32)

    ################################ TESTING ################################
    obs_test_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_test_masked.npy').astype(np.float32)

    #obs_final_train = tf.convert_to_tensor(obs_final_train,dtype=tf.float32)
    #obs_final_val = tf.convert_to_tensor(obs_final_val,dtype=tf.float32)
    #obs_final_test = tf.convert_to_tensor(obs_final_test,dtype=tf.float32)
    
    return(obs_train_RZSM,obs_val_RZSM,obs_test_RZSM)

def load_verification_observations_residuals(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    obs_train_RZSM = np.load(f'{verification_directory}/reforecast_residual_GEFSv12_lead_{lead}_train_masked.npy').astype(np.float32)

    ################################ VALIDATION ################################
    obs_val_RZSM = np.load(f'{verification_directory}/reforecast_residual_GEFSv12_lead_{lead}_val_masked.npy').astype(np.float32)

    ################################ TESTING ################################
    obs_test_RZSM = np.load(f'{verification_directory}/reforecast_residual_GEFSv12_lead_{lead}_test_masked.npy').astype(np.float32)

    #obs_final_train = tf.convert_to_tensor(obs_final_train,dtype=tf.float32)
    #obs_final_val = tf.convert_to_tensor(obs_final_val,dtype=tf.float32)
    #obs_final_test = tf.convert_to_tensor(obs_final_test,dtype=tf.float32)
    
    return(obs_train_RZSM,obs_val_RZSM,obs_test_RZSM)

def load_verification_observations_updated_TIMESERIES_CLASS(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    obs_train_RZSM = np.load(f'{verification_directory}/obs_RZSM_standardized_GLEAM_lead_{lead}_train_masked.npy').astype(np.float32)

    ################################ VALIDATION ################################
    obs_val_RZSM = np.load(f'{verification_directory}/obs_RZSM_standardized_GLEAM_lead_{lead}_val_masked.npy').astype(np.float32)

    ################################ TESTING ################################
    obs_test_RZSM = np.load(f'{verification_directory}/obs_RZSM_standardized_GLEAM_lead_{lead}_test_masked.npy').astype(np.float32)

    #obs_final_train = tf.convert_to_tensor(obs_final_train,dtype=tf.float32)
    #obs_final_val = tf.convert_to_tensor(obs_final_val,dtype=tf.float32)
    #obs_final_test = tf.convert_to_tensor(obs_final_test,dtype=tf.float32)
    
    return(obs_train_RZSM,obs_val_RZSM,obs_test_RZSM)

def load_verification_observations_updated_mean(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    obs_train_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_train_mean_masked.npy').astype(np.float32)

    ################################ VALIDATION ################################
    obs_val_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_val_mean_masked.npy').astype(np.float32)

    ################################ TESTING ################################
    obs_test_RZSM = np.load(f'{verification_directory}/obs_soilw_bgrnd_{obs_source}_lead_{lead}_test_mean_masked.npy').astype(np.float32)

    #obs_final_train = tf.convert_to_tensor(obs_final_train,dtype=tf.float32)
    #obs_final_val = tf.convert_to_tensor(obs_final_val,dtype=tf.float32)
    #obs_final_test = tf.convert_to_tensor(obs_final_test,dtype=tf.float32)
    
    return(obs_train_RZSM,obs_val_RZSM,obs_test_RZSM)



def load_verification_observations_grid_cell_standardization(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    #obs_train_tmax = np.load(f'{verification_directory}/OBS_tmax_ERA5_lead_{lead}_train_masked_grid_cell_standardization.npy').astype(np.float32)
    obs_train_RZSM = np.load(f'{verification_directory}/OBS_RZSM_GLEAM_lead_{lead}_train_masked_grid_cell_standardization.npy').astype(np.float32)

    #Now combined the train datasets for observations Verification and loss computation
    #obs_final_train = np.zeros(shape=(obs_train_tmax.shape[0],obs_train_tmax.shape[1],obs_train_tmax.shape[2],2)).astype(np.float32)
    #obs_final_train[:,:,:,0],obs_final_train[:,:,:,1] = obs_train_RZSM, obs_train_tmax
    
    ################################ VALIDATION ################################
    #obs_val_tmax = np.load(f'{verification_directory}/OBS_tmax_ERA5_lead_{lead}_val_masked_grid_cell_standardization.npy').astype(np.float32)
    obs_val_RZSM = np.load(f'{verification_directory}/OBS_RZSM_GLEAM_lead_{lead}_val_masked_grid_cell_standardization.npy').astype(np.float32)

    #Now combined the validation datasets for observations Verification and loss computation
    #obs_final_val = np.zeros(shape=(obs_val_tmax.shape[0],obs_val_tmax.shape[1],obs_val_tmax.shape[2],2))
    #obs_final_val[:,:,:,0],obs_final_val[:,:,:,1] = obs_val_RZSM, obs_val_tmax
    
    ################################ TESTING ################################
    #obs_test_tmax = np.load(f'{verification_directory}/OBS_tmax_ERA5_lead_{lead}_test_masked_grid_cell_standardization.npy').astype(np.float32)
    obs_test_RZSM = np.load(f'{verification_directory}/OBS_RZSM_GLEAM_lead_{lead}_test_masked_grid_cell_standardization.npy').astype(np.float32)

    #Now combined the validation datasets for observations Verification and loss computation
    #obs_final_test = np.zeros(shape=(obs_test_tmax.shape[0],obs_test_tmax.shape[1],obs_test_tmax.shape[2],2))
    #obs_final_test[:,:,:,0],obs_final_test[:,:,:,1] = obs_test_RZSM, obs_test_tmax
    
    #obs_final_train = tf.convert_to_tensor(obs_final_train,dtype=tf.float32)
    #obs_final_val = tf.convert_to_tensor(obs_final_val,dtype=tf.float32)
    #obs_final_test = tf.convert_to_tensor(obs_final_test,dtype=tf.float32)
    
    return(obs_train_RZSM,obs_val_RZSM,obs_test_RZSM)


def load_training_verification_observations(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    ################################ TRAINING ################################
    obs_train_tmax = np.load(f'{verification_directory}/OBS_tmax_ERA5_lead_{lead}_train_masked.npy').astype(np.float32)
    obs_train_RZSM = np.load(f'{verification_directory}/OBS_RZSM_GLEAM_lead_{lead}_train_masked.npy').astype(np.float32)

    #Now combined the train datasets for observations Verification and loss computation
    obs_final_train = np.zeros(shape=(obs_train_tmax.shape[0],obs_train_tmax.shape[1],obs_train_tmax.shape[2],2)).astype(np.float32)
    obs_final_train[:,:,:,0],obs_final_train[:,:,:,1] = obs_train_RZSM, obs_train_tmax

    return(obs_final_train)

def load_validation_verification_observations(lead,verification_directory):
    #For week 1, the observation verifications are all the exact same. Channel 1 is the RZSM and channel 2 is the Tmax.
    print('\nNow loading Verification Data from Observations.\n')
    
    ################################ VALIDATION ################################
    obs_val_tmax = np.load(f'{verification_directory}/OBS_tmax_ERA5_lead_{lead}_val_masked.npy').astype(np.float32)
    obs_val_RZSM = np.load(f'{verification_directory}/OBS_RZSM_GLEAM_lead_{lead}_val_masked.npy').astype(np.float32)

    #Now combined the validation datasets for observations Verification and loss computation
    obs_final_val = np.zeros(shape=(obs_val_tmax.shape[0],obs_val_tmax.shape[1],obs_val_tmax.shape[2],2))
    obs_final_val[:,:,:,0],obs_final_val[:,:,:,1] = obs_val_RZSM, obs_val_tmax
    
    return(obs_final_val)

def load_loss_csv(lead,experiment_name,RZSM_or_Tmax_or_both,region_name):
    loss_dir = f'Losses_with_OBS/Wk_{lead}'
    if RZSM_or_Tmax_or_both == 'both':
        loss_file = pd.read_csv(f'{loss_dir}/Wk{lead}_{experiment_name}')
    else:
        loss_file = pd.read_csv(f'{loss_dir}/Wk{lead}_{experiment_name}_RZSM')
    
    if RZSM_or_Tmax_or_both == 'both':
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
        tmax_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('tmax')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
    else:
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]

    #Find the channel with the smallest loss
    min_RZSM = list(RZSM_mae_val_losses.min())
    min_RZSM_index = min_RZSM.index(min(min_RZSM))
    
    if RZSM_or_Tmax_or_both == 'both':
        #Find the channel with the smallest loss
        min_tmax = list(tmax_mae_val_losses.min())
        min_tmax_index = min_tmax.index(min(min_tmax))
        return(min_RZSM_index,min_tmax_index)
    else:
        return(min_RZSM_index,None)
    

def load_loss_csv_bias_correction(lead,experiment_name_out,RZSM_or_Tmax_or_both,region_name):
    loss_dir = f'Losses_with_OBS/{region_name}/Wk{lead}'

    loss_file = pd.read_csv(f'{loss_dir}/Wk{lead}_{experiment_name_out}')
    
    if RZSM_or_Tmax_or_both == 'both':
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
        tmax_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('tmax')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
    else:
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]

    #Find the channel with the smallest loss
    min_RZSM = list(RZSM_mae_val_losses.min())
    min_RZSM_index = min_RZSM.index(min(min_RZSM))
    
    if RZSM_or_Tmax_or_both == 'both':
        #Find the channel with the smallest loss
        min_tmax = list(tmax_mae_val_losses.min())
        min_tmax_index = min_tmax.index(min(min_tmax))
        return(min_RZSM_index,min_tmax_index)
    else:
        return(min_RZSM_index,None)
    
def load_loss_csv_bias_correction_grid_cell_standardization(lead,experiment_name,RZSM_or_Tmax_or_both):
    loss_dir = f'Losses_with_OBS/Wk_{lead}'

    loss_file = pd.read_csv(f'{loss_dir}/Wk{lead}_{experiment_name}')
    
    if RZSM_or_Tmax_or_both == 'both':
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
        tmax_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('tmax')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]
    else:
        RZSM_mae_val_losses = loss_file.loc[:, loss_file.columns[(loss_file.columns.str.contains('RZSM')) & (loss_file.columns.str.contains('mae')) & (loss_file.columns.str.contains('val'))]]

    #Find the channel with the smallest loss
    min_RZSM = list(RZSM_mae_val_losses.min())
    min_RZSM_index = min_RZSM.index(min(min_RZSM))
    
    if RZSM_or_Tmax_or_both == 'both':
        #Find the channel with the smallest loss
        min_tmax = list(tmax_mae_val_losses.min())
        min_tmax_index = min_tmax.index(min(min_tmax))
        return(min_RZSM_index,min_tmax_index)
    else:
        return(min_RZSM_index,None)

def min_max_scaling(file,max_,min_):
    return(file-min_)/(max_-min_)


def restrict_to_CONUS_bounding_box(file, CONUS_mask):
    try:
        file = file.sel(latitude=slice(CONUS_mask.Y.values[0],CONUS_mask.Y.values[-1])).sel(longitude=slice(CONUS_mask.X.values[0],CONUS_mask.X.values[-1]))
    except KeyError:
        file = file.sel(Y=slice(CONUS_mask.Y.values[0],CONUS_mask.Y.values[-1])).sel(X=slice(CONUS_mask.X.values[0],CONUS_mask.X.values[-1]))
    return(file)


def restrict_to_CONUS_bounding_box_only_04A_script(file, CONUS_mask):
    try:
        file = file.sel(latitude=slice(CONUS_mask.Y.values[0],CONUS_mask.Y.values[-1])).sel(longitude=slice(CONUS_mask.X.values[0],CONUS_mask.X.values[-1]))
    except KeyError:
        file = file.sel(Y=slice(CONUS_mask.Y.values[0],CONUS_mask.Y.values[-1])).sel(X=slice(CONUS_mask.X.values[0],CONUS_mask.X.values[-1]))
    return(file)

def make_additional_predictions_from_model_for_testing(lead,experiment_name,model,reforecast_testing_input,num_predictions_testing):
    single_prediction = model.predict(reforecast_testing_input) #Just for unit checking
    out_name_save = f'predictions/Wk{lead}_testing/{num_predictions_testing}_test_predictions'
    os.system(f'mkdir -p {out_name_save}')
    
    #if os.path.exists(out_name_save):
        #print(f'Already completed experiment {experiment_name} {num_predictions_testing} testing predictions')
    #else:
        #print(f'Making {num_predictions_testing} testing predictions for experiment {experiment_name}.')

    np.save(f'{out_name_save}/Wk{lead}_testing_{num_predictions_testing}_test_predictions_{experiment_name}', \
        np.concatenate([np.array(model.predict(reforecast_testing_input)) for _ in range(num_predictions_testing)],axis=-1).reshape(np.array(single_prediction).shape[0],104,11,48,96,num_predictions_testing))
    return(0)





def load_channel_list_permutation(experiment_name, lead):
    channel_file = f'channel_list_information/Wk{lead}/{experiment_name}_channel_list.txt'
    with open(channel_file, 'r') as file:
        # Read the contents of the file
        file_contents = file.read()
        file_contents = file_contents.split('\n')
        split_list = [sentence.split() for sentence in file_contents] #Completely split the sentence
        total_channel_number = len(split_list)
        
        channel_list = []
        for i in range(total_channel_number):
            try:
                channel_list.append(split_list[i][-1])
            except IndexError:
                pass
                #Sometimes the last line is just blank "". So this is a typo in the original script, no need to fix in the original because of runtime
    return(channel_list)




def load_min_max_files_and_rescale_data(testing_input,channel,idx,reforecast_source, region_name,day_num,test_year,lead):
    '''This function will take the current data that has been min-max scaled and then it will re-scale it back to anomalies. 
    Then we will add some gaussian noise and re-scale back to it's min-max form.'''        


    min_max_dir = f'Data/min_max_values/{region_name}'
    
    output_with_noise = np.array(testing_input)
                                 
    if 'ref' in channel or 'prediction' in channel:
        if 'ECMWF' in reforecast_source:
            source_ = 'ECMWF'
        else:
            source_ = 'GEFSv12'        
    elif 'obs' in channel:
        if ('RZSM' in channel) or ('soil' in channel):
            source_ = 'GLEAM'
        else:
            source_ = 'ERA5'
    
    if ('RZSM' in channel) or ('soil' in channel):
        #There are zero values for RZSM and this may mess up some of the calculations
        reforecast_nan = np.where(testing_input[:,:,:,idx]==0,np.nan,testing_input[:,:,:,idx])
    else:
        reforecast_nan = testing_input[:,:,:,idx]
     
    #Load min max data
    if 'diff_temp' in channel:
        var_ = 'diff_temp_2m'
    elif ('z200' in channel) or ('hgt_pres' in channel):
        var_ = 'hgt_pres'
    elif 'RZSM' in channel:
        var_ = 'soilw_bgrnd'
    elif 'pwat' in channel:
        var_ = 'pwat_eatm'
    elif 'tmax' in channel:
        var_ = 'tmax_2m'
    elif 'spfh' in channel:
        var_ = 'spfh_2m'
    else:
        var_ = channel.split('_')[0]
        
    print(f'Variable is {var_}')

    if test_year == 2019:
        max_ = np.load(f'{min_max_dir}/{var_}_{source_}_lead{day_num}_max.npy')
        min_ = np.load(f'{min_max_dir}/{var_}_{source_}_lead{day_num}_min.npy')
    else:
        max_ = np.load(f'{min_max_dir}/{var_}_{source_}_lead{day_num}_max_{test_year}.npy')
        min_ = np.load(f'{min_max_dir}/{var_}_{source_}_lead{day_num}_min_{test_year}.npy')

    '''Now we have the max and min values for each file, we need to reverse it back to anomaly'''
    de_normalized = verifications.reverse_min_max_scaling_for_permutations(reforecast_nan,region_name,day_num,source_,test_year,var_)
    
    # np.random.seed(0) #Set the seed just for more clarity in interpreting results
    '''Then we need to create gaussian noise'''
    noise = np.random.normal(np.nanmean(de_normalized), np.nanstd(de_normalized),de_normalized.shape)
    '''Add noise'''
    noisy_data = de_normalized + noise
    
    '''Save the new min max after noise has been added'''
    noise_max_anomaly = np.nanmax(noisy_data)
    noise_min_anomaly = np.nanmin(noisy_data)
    '''Min-max scale again, but with the new min and max after adding noise'''
    noisy_data = min_max_scaling(noisy_data,np.nanmax(noisy_data),np.nanmin(noisy_data))
    '''Then convert any np.nan values back to zero (this only applies if the variable was RZSM'''
    noisy_data = np.where(np.isnan(reforecast_nan),0,noisy_data)
    
    output_with_noise[:,:,:,idx] = noisy_data

    return(output_with_noise,reforecast_nan,noise_min_anomaly,noise_max_anomaly,var_)


def create_seasonal_anomaly(file,train_end,source):
    if source == 'reforecast':
        time_name = 'S'
    elif source == 'obs':
        time_name = 'time'
    
    #First create a climatology by season based on training data only (mean is coming directly from training data and applied to validation and testing)
    climpred.set_options(seasonality="season") 
    seasonality_str = OPTIONS["seasonality"]
    
    #change the dates to be pd.to_datetime objects
    file[f'{time_name}'] = pd.to_datetime(file[f'{time_name}'].values)
    
    if source == 'obs':
        '''need to have a separate section so that the observations "time" is set. Reforecasts have "S'''
        climatology_season = file.sel(time=(file[f'{time_name}.year'] <= train_end)).groupby(f"{time_name}.{seasonality_str}").mean()

        summer_= file.sel(time=(file[f'{time_name}.season']=='JJA')) - climatology_season.sel(season='JJA')
        fall_= file.sel(time=(file[f'{time_name}.season']=='SON'))- climatology_season.sel(season='SON')
        winter_= file.sel(time=(file[f'{time_name}.season']=='DJF'))- climatology_season.sel(season='DJF')
        spring_= file.sel(time=(file[f'{time_name}.season']=='MAM'))- climatology_season.sel(season='MAM')
    elif source == 'reforecast':
        climatology_season = file.sel(S=(file[f'S.year'] <= train_end)).groupby(f"S.{seasonality_str}").mean()

        summer_= file.sel(S=(file[f'{time_name}.season']=='JJA')) - climatology_season.sel(season='JJA')
        fall_= file.sel(S=(file[f'{time_name}.season']=='SON'))- climatology_season.sel(season='SON')
        winter_= file.sel(S=(file[f'{time_name}.season']=='DJF'))- climatology_season.sel(season='DJF')
        spring_= file.sel(S=(file[f'{time_name}.season']=='MAM'))- climatology_season.sel(season='MAM')

    combined_files = xr.concat([summer_,fall_,winter_,spring_],dim=f'{time_name}').sortby(time_name)
    combined_files = combined_files.drop('season')
    
    
    return(combined_files) #combine all anomalies, sort by date


def convert_prediction_to_SubX_format(file,lead,dim_order,region_name,base_file_testing):

    if region_name == 'CONUS':
        cp_base = f.restrict_to_CONUS_bounding_box(base_file_testing.copy(deep=True).sel(L=(lead*7)-1),mask).expand_dims({'L': 1})
    else:
        cp_base = base_file_testing.copy(deep=True).sel(L=(lead*7)-1).expand_dims({'L': 1})

    
    #reshape data back to original format (testing shape)
    
    cp_base = cp_base.transpose(*dim_order)
    
    #Reshape prediction file
    file = file.reshape((104,11,1,48,96))
    
    var_OUT = xr.Dataset(
            data_vars = dict(
                file_name = (['S','M','L','Y','X'],  file[:,:,:,:,:]),
            ),
            coords = dict(
                S = cp_base.S.values,
                X = cp_base.X.values,
                Y = cp_base.Y.values,
                L = cp_base.L.values,
                M = cp_base.M.values,

            ),
            attrs = dict(
                Description = 'New data added to file'),
        )  

    return(var_OUT)
