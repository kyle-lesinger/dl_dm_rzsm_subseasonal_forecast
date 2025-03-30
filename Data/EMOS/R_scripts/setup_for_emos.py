#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:24:04 2023

Create a dataframe object that can be opened in R for ensemble model output
statistics processing.

Need to save in the following format
[1] "idate"         "vdate"         "latitude"      "longitude"     "region"            
   "T2.obs"        "T2.gfs"        "T2.cmcg"       "T2.eta"        "T2.gasp"      
[13] "T2.jma"        "T2.ngps"       "T2.tcwb"       "T2.ukmo"    

But instead of T2, we have RZSM.0, RZSM.1, through RZSM.11 for 11 different models

And our region is actually going to be the lat/lon values

Let's just setup a single dataframe. We are only trying to do root-zone soil
moisture.

We are trying to improve the prediction of anomalies
@author: kdl
"""

import xarray as xr
import numpy as np
import os
import pandas as pd
from glob import glob
from multiprocessing import Pool
import datetime as dt
import climpred
from climpred.options import OPTIONS
import config_FD

seagate_GEFS_dir = config_FD.reanalysis_media_dir
dir1= config_FD.home_dir

##################### REFORECAST #############################
data_dir = f'{dir1}/Data/EMC_reforecast'

save_csv_dir = f'{dir1}/Data/EMOS'

RZSM_dir = f'{data_dir}/soilw_bgrnd'

#find the list of available dates
get_last_date_of_reanalysis = xr.open_mfdataset(f'{dir1}/Data/GLEAM_v3.7a/RZSM_reformat_to_reforecast_shape/RZSM*')
get_last_date_of_reanalysis=get_last_date_of_reanalysis.rename(Y='latitude',X='longitude',M='model', S = 'idate', L='vdate')
#NCA_LDAS mask. 0s if a water body, 1 if land
mask_file_original = xr.open_dataset(f'{dir1}/Data/CONUS_masks_grids/NCA-LDAS_masks_0.5degree_grid.nc4')
mask_file_subx = mask_file_original['NCA-LDAS_mask'].rename(longitude='X').rename(latitude='Y').rename(time='S')


def return_name_of_xarray_var(file):
    return(list(file.keys())[0])

#other lists
rzsm_list = glob(f'{RZSM_dir}/*RZSM_mean*0-100cm*')
open_files =xr.open_mfdataset(rzsm_list)
open_files = open_files.where(mask_file_subx[0,:,:] == 1) #mask ocean and water bodies with np.nan

#Restrict size of files to match UNET size of input
open_files = open_files[return_name_of_xarray_var(open_files)].isel(Y=slice(0,48)).isel(X=slice(6,len(open_files.X.values)-6)).to_dataset().sel(S=get_last_date_of_reanalysis.idate.values)

open_files=open_files.rename(Y='latitude',X='longitude',M='model', S = 'idate', L='vdate')


''''We don't have to create a new training and testing set, we will just include them in the same dataframe
but we do need to know how many dates there were

939 training dates
104 testing dates
'''

##################### REANALYSIS (OBSERVATIONS) #############################
#RZSM obs has already had the 7-day running mean applied
obs_dir = f'{dir1}/Data/GLEAM_v3.7a'
# RZSM_obs =  xr.open_dataset(f'{obs_dir}/RZSM_0_100cm.nc4')
# #restrict CONUS bounding box
output_reanalysis_reformat_dir = f'{dir1}/Data/GLEAM_v3.7a/RZSM_reformat_to_reforecast_shape'
RZSM_obs = xr.open_mfdataset(f'{output_reanalysis_reformat_dir}/RZSM*')
RZSM_obs = RZSM_obs.sel(S=get_last_date_of_reanalysis.idate.values)
RZSM_obs=RZSM_obs.rename(Y='latitude',X='longitude',M='model', S = 'idate', L='vdate')
RZSM_obs=RZSM_obs.assign_coords(idate=open_files.idate.values) #re-assign dates 

# RZSM_obs.RZSM[10,0,0,20,10].values


def create_seasonal_anomaly(file):
    #First create a climatology by season
    climpred.set_options(seasonality="season") 
    seasonality_str = OPTIONS["seasonality"]
    climatology_season = file.groupby(f"idate.{seasonality_str}").mean()
    
    summer_=file.sel(idate=(file['idate.season']=='JJA')) - climatology_season.sel(season='JJA')
    fall_=file.sel(idate=(file['idate.season']=='SON'))- climatology_season.sel(season='SON')
    winter_=file.sel(idate=(file['idate.season']=='DJF'))- climatology_season.sel(season='DJF')
    spring_=file.sel(idate=(file['idate.season']=='MAM'))- climatology_season.sel(season='MAM')


    return(xr.concat([summer_,fall_,winter_,spring_],dim='idate').sortby('idate')) #combine all anomalies, sort by date
    # combine_all_data = combine_all_data.sortby('S') #sort data by date
    


save_obs = f'{config_FD.gleam_dir}/RZSM_anomaly_0_100cm_SubX_format.nc4'
save_subx = f'{dir1}/Data/EMC_reforecast/RZSM_anomaly_0_100cm.nc4'

try:
    obs_anomaly = xr.open_dataset(save_obs)
    subx_anomaly = xr.open_dataset(save_subx)
except FileNotFoundError:    
    #Now remove the seasonal mean to create the anomaly
    #Don't re-create the file if already completed
    obs_anomaly_all =  create_seasonal_anomaly(file=RZSM_obs)
    subx_anomaly = create_seasonal_anomaly(file=open_files)
    
    #apply a 7-day rolling mean to be consistent with the other data
    obs_anomaly=obs_anomaly_all.rolling(vdate=7, min_periods=7,center=False).mean()
    subx_anomaly=subx_anomaly.rolling(vdate=7, min_periods=7,center=False).mean()
    
    #But its important that we re-add the day 0 obs_anomaly for future values
    obs_anomaly.RZSM[:,:,0,:,:] = obs_anomaly_all.RZSM[:,:,0,:,:]
    
    '''add mask because EMOS is having issue with no data
    obs_anomaly=obs_anomaly.where(mask_file_original['NCA-LDAS_mask'][0,:,:]==1)
    #then add a mask to SubX files to make sure that everything is np.nan
    subx_anomaly=subx_anomaly.where(obs_anomaly != np.nan,subx_anomaly,np.nan)
    #save files for later use
    
    NOTE: Not doing this anymore, we are just going to remove the data in R (within EMOS script)
    by removing the np.nan values. I can't quite get the mask to work out perfectly'''
    
    
    
    
    obs_anomaly.to_netcdf(save_obs)
    subx_anomaly.to_netcdf(save_subx)
    
    #Now re-open the files that were saved because they don't load into memory with climpred 
    #and processing the files takes forever.
    obs_anomaly = xr.open_dataset(save_obs)
    subx_anomaly = xr.open_dataset(save_subx)
#Just rename the files to work with below functions
# RZSM_obs=obs_anomaly
# open_files=subx_anomaly
#%% Work on leads 1,2,3,4 (weekly leads)
#stack models together in a certain format (listed at top of script)

def return_combined_models(init_day,lead):
    all_models={}
    # obs = {}
    for model in range(len(subx_anomaly.model.values)):
        model_val = subx_anomaly.RZSM.sel(idate=init_day,vdate=lead).isel(model=model).to_dataframe().reset_index()
        model_val = model_val.rename(columns={'RZSM':f'RZSM.{model}'}) #rename RZSM column
        del model_val['model']
        all_models[f'Model{model}'] = model_val
    
    #add observations, just use a single model because we reformatted the data in a previous script
    obs_val = obs_anomaly.RZSM.sel(idate=init_day,vdate=lead).isel(model=0).to_dataframe().reset_index()
    obs_val = obs_val.rename(columns={'RZSM':'RZSM.obs'}) #rename RZSM column
    # del obs_val['model']
    # obs['obs'] = obs_val
    
    #Now lets just add new columns to each all_models['Model0'] (joining doesn't work very well)
    for model in range(1,11):
        #there are technically 11 models, but we already have the 0th model in the dataset
        all_models['Model0'][f'RZSM.{model}']= all_models[f'Model{model}'][f'RZSM.{model}']

    #Convert the vdate (verification date)
    all_models['Model0']['vdate'] = all_models['Model0']['idate'] + np.timedelta64(lead+1,'D') #Must add 1 because of the indexing with python to get the correct date from GLEAM observations
    all_models['Model0']['RZSM.obs']=obs_val['RZSM.obs']
    
    return(all_models['Model0'])

#Now combine all models by day
def iterate_over_all_days(lead):
    print(f'Working on lead {lead}')
    # num_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine','10':'ten'}

    out_df = pd.DataFrame()
    for idx,date in enumerate(subx_anomaly.idate.values):
        # if lead==6:
            # print(f'Working on init {date}.')
        # break
        out_df = pd.concat([out_df,return_combined_models(init_day=date,lead=lead)],axis=0)
    # out_df = [out_df.rename(columns={f'RZSM.{i}':f'RZSM.{list(num_dict.values())[i]}'}) for i in range(len(num_dict))]
    out_df = out_df.reset_index()
    out_df['lat_lon'] = out_df["latitude"].astype(str) +"-"+ out_df["longitude"].astype(str) #combine lat and lon
    del out_df['index']
    
    #convert dates to format yearmonthday00
    all_dates=pd.to_datetime(out_df['vdate']).dt.date
    all_dates=list(all_dates)
    all_dates = ['{:%Y%m%d}00'.format(i) for i in all_dates]
    out_df['vdate'] = all_dates
    
    #Now choose the train, validation and testing days
    training = out_df[out_df['idate'] <= '2015-12-31'].reset_index(drop=True)
    validation = out_df[out_df['idate'] <= '2017-12-31']
    validation = validation[validation['idate'] >='2016-01-01'].reset_index(drop=True)
    testing = out_df[out_df['idate'] >='2018-01-01'].reset_index(drop=True)
    
    
    out_df.to_csv(f'{save_csv_dir}/emos_setup_train_val_test_lead{lead+1}.csv.gz',compression='gzip') #save as gzip to reduce storage #all data, no split train,val,test
    training.to_csv(f'{save_csv_dir}/emos_training_setup_lead{lead+1}.csv.gz',compression='gzip')
    validation.to_csv(f'{save_csv_dir}/emos_validation_setup_lead{lead+1}.csv.gz',compression='gzip')
    testing.to_csv(f'{save_csv_dir}/emos_testing_setup_lead{lead+1}.csv.gz',compression='gzip')
    
    return(0)    



# for lead in [0,6,13,20,27,34]:
#     iterate_over_all_days(lead)
#%%

if __name__ == '__main__':
    p=Pool(5)
    p.map(iterate_over_all_days,[6,13,20,27,34])
    
    
    
# week1_df = iterate_over_all_days(lead=7)
# week2_df = iterate_over_all_days(lead=14)  
# week3_df = iterate_over_all_days(lead=21)
# week4_df = iterate_over_all_days(lead=28)


# train_set_dates = open_files.sel(S=(open_files['S.year'] <= 2017)) 
# test_set = open_files.sel(S=(open_files['S.year'] >= 2018))


#%% Work only on lead 0, just reopen files without doing a 7-day rolling mean
#stack models together in a certain format (listed at top of script)


# #other lists
# rzsm_list = glob(f'{RZSM_dir}/*RZSM_mean*0-100cm*')
# open_files =xr.open_mfdataset(rzsm_list)
# open_files = open_files.where(mask_file[0,:,:] == 1) #mask ocean and water bodies with np.nan

# #Restrict size of files to match UNET size of input (do not apply a 7-day rolling mean) becasue then there will be no value for index 0
# open_files = open_files[return_name_of_xarray_var(open_files)].isel(Y=slice(0,48)).isel(X=slice(6,len(open_files.X.values)-6)).to_dataset().sel(S=get_last_date_of_reanalysis.idate.values)


# open_files=open_files.rename(Y='latitude',X='longitude',M='model', S = 'idate', L='vdate')


# iterate_over_all_days(lead=0)