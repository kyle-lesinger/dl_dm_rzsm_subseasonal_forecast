#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 08:25:51 2023

ToDo: reload data and make sure we have the correct init days


@author: kdl
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import os
import pandas as pd
from glob import glob
import sys
import datetime as dt
from xarray import apply_ufunc
from numba import prange,njit
import config_FD



#potential different test names
test_all = ['all_training','all_training_and_validation','12_weeks_before',
            '12_weeks_before_and_after']
# test_all = ['12_weeks_before']

num_predictions= [11,100]
#%%
def emos_different_training_tests(lead,test_,emos_dir,n_predictions):
    if test_ == 'all_training':
        emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_training_dataset.csv') 
        out_name = f'{emos_dir}/EMOS_{n_predictions}_test_predictions_from_all_training.nc'
    elif test_ == '12_weeks_before_and_after':
        emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_training_dataset_12_weeks_before_and_after.csv')
        out_name = f'{emos_dir}/EMOS_{n_predictions}_test_predictions_12_weeks_before_and_after.nc'
    elif test_ == 'all_training_and_validation':
        emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_training_and_validation_dataset.csv')
        out_name = f'{emos_dir}/EMOS_{n_predictions}_test_predictions_from_all_training_validation.nc'
    elif test_ == '12_weeks_before':
        emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_12_previous_weeks_across_all_training_data.csv')
        out_name = f'{emos_dir}/EMOS_{n_predictions}_test_predictions_12_weeks_before.nc'
        
    return(emos_file, out_name)


#In case there are errors in preprocessing (making duplicate files)
def drop_duplicates_along_all_dims(obj, keep="first"):
   all_dims = obj.dims
   indexes = {dim: ~obj.get_index(dim).duplicated(keep=keep) for dim in all_dims}
   return obj.isel(indexes)

print(f"PYTHON: {sys.version}")  # PYTHON: 3.8.1 | packaged by conda-forge | (default, Jan 29 2020, 15:06:10) [Clang 9.0.1 ]
print(f" xarray {xr.__version__}")  # xarray 0.14.1
print(f" numpy {np.__version__}")  # numpy 1.17.3
print(f" matplotlib {mpl.__version__}")  # matplotlib 3.1.2

# TODO change later
dir1=config_FD.home_dir

emos_dir=config_FD.emos_dir

#Now load the original files (GLEAM) so that we can re-create a netcdf file
output_reanalysis_reformat_dir = config_FD.gleam_dir
obs_RZSM_files = xr.open_dataset(f'{output_reanalysis_reformat_dir}/RZSM_anomaly_0_100cm_SubX_format.nc4')
dates_list = list(obs_RZSM_files.idate.values)
df = pd.DataFrame({'dates':dates_list})
df['dates'] = pd.to_datetime(df['dates'])
df['dates'] = df['dates'].dt.floor('d')
new_date_list = np.array(df['dates'])
obs_RZSM_files=obs_RZSM_files.assign_coords(idate=new_date_list) #add all data to this

obs_RZSM_files = obs_RZSM_files.where(obs_RZSM_files==8675309) #just convert all values to np.nan to make sure we have only emos values

#We can see later from opening up the emos file that we are only looking at specific days, it needed 12 weeks for training
obs_RZSM_files = obs_RZSM_files.sel(idate=slice('2017-12-27','2019-12-25'))

#Make new variables for the values 
# test_set_full = test_set_full.assign(mae=test_set_full['RZSM'],mse=test_set_full['RZSM'],rmse=test_set_full['RZSM'],ensemble_crps=test_set_full['RZSM'],EMOS_crps=test_set_full['RZSM'],stddev=test_set_full['RZSM'])
#These are all the variables from the emos file in the R script
obs_RZSM_files = obs_RZSM_files.assign(mae=obs_RZSM_files['RZSM'],mse=obs_RZSM_files['RZSM'],\
                                     rmse=obs_RZSM_files['RZSM'],ensemble_crps=obs_RZSM_files['RZSM'],\
                                         emos_crps=obs_RZSM_files['RZSM'],emos_mean_std=obs_RZSM_files['RZSM'],\
                                             emos_mean_prediction = obs_RZSM_files['RZSM'],
                                             observation = obs_RZSM_files['RZSM'])


original_OBS = obs_RZSM_files['RZSM'].sel(vdate=[6,13,20,27,34]).sel(idate=slice('2018-01-03','2019-12-25')) #Used to re-add data after creating disribution
#Only save 1 model for inputting the data
obs_RZSM_files = obs_RZSM_files.drop('RZSM').sel(vdate=[6,13,20,27,34]).isel(model=0)

#Must get only the year 2018
obs_RZSM_files = obs_RZSM_files.sel(idate=slice('2018-01-03',obs_RZSM_files.idate.values[-1]))


#%%
sys.setrecursionlimit(10000000) 
#run through different EMOS tests
for n_predictions in num_predictions:
    for test_ in test_all:
    
        #Add the data to a common dataset for later processing
        #Get the lat and lon values, then add the data into a netcdf file by date
        for idx_lead,lead in enumerate([7,14,21,28,35]):
            print(f'Working on lead {lead}')
            # lead = 7
        
            #Load an EMOS completed file (different testing configurations)
            #this is from the entire training dataset ~831 weeks to train on 
            emos_file, out_name = emos_different_training_tests(lead,test_,emos_dir,n_predictions)

            # emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_training_dataset_12_weeks_before_and_after.csv')
            # emos_file = pd.read_csv(f'{emos_dir}/emos_completed_lead{lead}_test_predictions_only_from_training_and_validation_dataset.csv')
        
            #convert lat/lon to new columns
            lat_lon = pd.DataFrame(emos_file['lat_longitude'])
            
            lat=[]
            lon=[]
            for i in lat_lon.iloc[:,0]:
                lat.append(i.split('-')[0])
                lon.append(i.split('-')[1])
            
            emos_file['lat'] = lat
            emos_file['lon'] = lon
            
            lon_float = emos_file['lon'].astype(float)
            emos_file['lon']=lon_float
            lat_float = emos_file['lat'].astype(float)
            emos_file['lat']=lat_float
            
            del emos_file['lat_longitude']
            
            df_multiindex = emos_file.set_index(['lon', 'lat','init'])
            df_array = df_multiindex.to_xarray()
            df_array = df_array.drop('observation_day')
            
            #Fix some date issues
            df_array
            all_dates = [pd.to_datetime(i) for i in df_array.init.values]
            date_vals = pd.to_datetime(all_dates)
            index_start = np.where(date_vals ==pd.to_datetime('2018-01-03'))[0][0]
            all_dates = all_dates[index_start:]
            
            #get index of init '2018-01-03'
            date_vals = pd.to_datetime(df_array.init.values)
            index_start = np.where(date_vals ==pd.to_datetime('2018-01-03'))[0][0]
            df_array = df_array.isel(init = slice(index_start,df_array.init.shape[0]))
            
            df_array = df_array.assign_coords({'init': all_dates})
            #must reverse the order of latitude
            df_array = df_array.isel(lat=slice(None, None, -1))
            
            df_array = df_array.transpose("init", "lat", "lon")
        
            #remove duplicates
            df_array = drop_duplicates_along_all_dims(df_array)

        
            print(f'init length: {df_array.init.shape[0]}')
            
            for var in obs_RZSM_files.keys():
                # print(var)
                #Some dates are not always available for each lead
                avail_dates = obs_RZSM_files.sel(idate=all_dates)
                avail_dates[var][:,idx_lead,:,:] = df_array[var][:,:,:]
                
                if lead == 0:
                    lead_sel = 0
                else:
                    lead_sel=lead-1
                
                obs_RZSM_files[var].loc[dict(idate=all_dates,vdate=lead_sel)] = df_array[var][:,:,:]
    

#%%
        '''Now that we have added all the data back from EMOS, we need to use the predicted
        mean and predicted std to create 11 ensemble forecasts from a normal distribution'''
        
        #create a blank document
        input_ = original_OBS.to_numpy().squeeze() 
        final_output = original_OBS.copy(deep=True)
        input_.shape
        
        
        obs_RZSM_files.rmse.shape
        
        mean_ = obs_RZSM_files['emos_mean_prediction'].to_numpy()
        std_ = obs_RZSM_files['emos_mean_std'].to_numpy()
        
        mean_.shape
        std_.shape
        input_.shape
        # final_output.observation.shape
    
    
        #using only 11 predictions drawn from the distibution
        @njit
        def add_prediction_from_distribution(input_file,mean_,std_,n_predictions):
            output_ = np.empty_like(input_file)
            for day in prange(input_file.shape[0]):
                for lead in range(input_file.shape[2]):
                    for Y in range(input_file.shape[3]):
                        for X in range(input_file.shape[4]):
                            # break
                        
                            
                            loc = mean_[day,lead,Y,X]
                            scale = std_[day,lead,Y,X]
                            #Now draw from the gaussian distribution
                            output_[day,:,lead,Y,X] = np.random.normal(loc=loc,scale=scale,size=n_predictions)
                            
                            # elif lead==1:
                            #     if day == 0:
                            #         #Now draw from the gaussian distribution
                            #         output_[day,:,lead,Y,X] = np.nan
                            #     elif day != 0:
                            #         #Now draw from the gaussian distribution
                            #         #There was an issue with 1 date from lead 7 (the very first date)
                            #         loc = mean_[day,lead,Y,X]
                            #         scale = std_[day,lead,Y,X]
                            #         output_[day,:,lead,Y,X] = np.random.normal(loc=loc,scale=scale,size=num_predictions)
                            
            return(output_[:,:,:,:,:])
        
        if n_predictions == 11:
            emos_completed = add_prediction_from_distribution(input_file=input_,mean_=mean_,std_=std_,n_predictions=n_predictions)
            
            final_output[:,:,:,:,:] = emos_completed[:,:,:,:,:]
            
            
            final_output = final_output.to_dataset()
            final_output.to_netcdf(f'{out_name}')
        else:
            
            def create_predictions_and_save_greater_than_11(n_predictions):
                #create a blank dataset
                input_ = np.empty((original_OBS.shape[0],n_predictions,original_OBS.shape[2],original_OBS.shape[3],original_OBS.shape[4]))
                
                #Now re-run function from above
                emos_completed = add_prediction_from_distribution(input_file=input_,mean_=mean_,std_=std_,n_predictions=n_predictions)
                
                
                final_output = xr.Dataset(
                    data_vars = dict(
                        RZSM = (['idate', 'model','vdate','latitude','longitude'], emos_completed),
                    ),
                    coords = dict(
                        longitude = original_OBS.longitude.values,
                        latitude = original_OBS.latitude.values,
                        vdate = original_OBS.vdate.values,
                        model = range(emos_completed.shape[1]),
                        idate = original_OBS.idate.values
                    ),
                    attrs = dict(
                        Description = 'EMOS predictions.'),
                )   
                
                final_output.to_netcdf(f'{out_name}')
                
                return(0)
            
            
            create_predictions_and_save_greater_than_11(n_predictions)

#%%

