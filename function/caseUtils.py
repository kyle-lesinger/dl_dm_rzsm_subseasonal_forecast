#!/usr/bin/env python3
import xarray as xr
import numpy as np
import climpred
from xclim import sdba
from glob import glob
import os
import random
import pandas as pd

from function import funs as f
from function import preprocessUtils as putils
from function import masks
from function import conf




def return_short_array(path,test_start,test_end):
    y1 = str(pd.to_datetime(test_start).year)
    y2 = str(pd.to_datetime(test_end).year)

    outpaths = []
    for p in sorted(glob(path)):
        if (y1 in p) or (y2 in p):
            outpaths.append(p)
    return(outpaths)

def open_obs_and_baseline_files(region_name, week_lead, day_num, start_, end_, mask_anom, test_start, test_end, obs_source,soil_dir):

    print(f'Loading template data from {test_start} to {test_end}')
    mask,mask_anom = masks.load_mask_vals(region_name)

    obs_anomaly_SubX_format_testing =xr.open_mfdataset(return_short_array(f'{soil_dir}/{region_name}/RZSM_anomaly_reformat_SubX_format/RZSM_anomaly*',test_start,test_end)).sel(L=[day_num]).load()

    template_testing_only = obs_anomaly_SubX_format_testing.copy(deep=True)

    assert pd.to_datetime(start_).year == pd.to_datetime(end_).year, 'For this code to work properly, you must have the beginning and ending of the case study in the same year!'

    print(f'Loading {obs_source}, gefsv12 raw reforecast, and ecmwf raw reforecast during case study dates of {start_} to {end_}')


    var_OUT = np.empty(shape=(obs_anomaly_SubX_format_testing.Y.shape[0], obs_anomaly_SubX_format_testing.X.shape[0])) #48x96
    var_OUT[:,:] = 0
    
    #Mask the final output to be np.nan for ocean values
    if np.count_nonzero(np.isnan(mask_anom))>=2:
        var_OUT = np.where(mask_anom==1, np.nan, var_OUT)
    else:
        var_OUT = np.where(np.isnan(mask_anom), np.nan, var_OUT)

    
    #######################################   Reforecast baseline files   ###########################################################################
    # baseline_anomaly_file_list = sorted(glob('Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/RZSM*.nc'))
    if region_name =='CONUS':
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    
    else:
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)
    
    baseline_ecmwf_file_list = sorted(glob(f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    ecmwf_bias_corrected = xr.open_dataset(conf.return_bias_corrected_anomaly(region_name, 'ECMWF', obs_source)).isel(lead=[6,13,20,27,34]).rename({'init':'S','member':'M','lead': 'L','lat':'Y','lon':'X'})
    gefs_bias_corrected = xr.open_dataset(conf.return_bias_corrected_anomaly(region_name, 'GEFSv12', obs_source)).isel(lead=[6,13,20,27,34]).rename({'init':'S','member':'M','lead': 'L','lat':'Y','lon':'X'})

    ecmwf_bias_corrected['L'] = [6,13,20,27,34]
    gefs_bias_corrected['L'] = [6,13,20,27,34]
    
    ecmwf_bias_corrected=ecmwf_bias_corrected.sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    gefs_bias_corrected=gefs_bias_corrected.sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format_testing, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only,ecmwf_bias_corrected,gefs_bias_corrected)


def open_only_testing_anomaly_baseline_files(region_name, week_lead, day_num, mask_anom, test_start, test_end):

    print(f'Loading template data from {test_start} to {test_end}')
    mask,mask_anom = load_mask(region_name)
    
    obs_anomaly_SubX_format_testing =xr.open_mfdataset(return_short_array(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*',test_start,test_end)).sel(L=[day_num]).load()
    template_testing_only = obs_anomaly_SubX_format_testing.copy(deep=True)

    assert pd.to_datetime(start_).year == pd.to_datetime(end_).year, 'For this code to work properly, you must have the beginning and ending of the case study in the same year!'

    print(f'Loading observation, gefsv12 raw reforecast, and ecmwf raw reforecast during case study dates of {start_} to {end_}')
    obs_anomaly_SubX_format =xr.open_mfdataset(f'Data/GLEAM/RZSM_anomaly_reformat_SubX_format/{region_name}/RZSM_anomaly*{pd.to_datetime(start_).year}*.nc4').sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    var_OUT = np.empty(shape=(obs_anomaly_SubX_format.Y.shape[0], obs_anomaly_SubX_format.X.shape[0])) #48x96
    
    #Mask the final output to be np.nan for ocean values
    var_OUT = np.where(mask_anom==1, np.nan, var_OUT)
    var_OUT[:,:] = 0
    
    #######################################   Reforecast baseline files   ###########################################################################
    # baseline_anomaly_file_list = sorted(glob('Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/RZSM*.nc'))
    if region_name =='CONUS':
        baseline_anomaly_file_list = sorted(glob(f'Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
    
    else:
        baseline_anomaly_file_list = sorted(glob(f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)
    
    baseline_ecmwf_file_list = sorted(glob(f'Data/ECMWF/soilw_bgrnd_processed/{region_name}/baseline_RZSM_anomaly/soil*{pd.to_datetime(start_).year}*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=[day_num]).sel(S=slice(start_,end_)).load()

    
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only)



def return_case_study_dates(region_name,test_year):
    #dates for flash drought event
    if region_name == 'CONUS':
        if test_year == 2019:
            start_ = '2019-08-21'
            end_ = '2019-09-25'
    
        elif test_year == 2012:
            start_ = '2012-05-01'
            end_ = '2012-07-15'
     
            
        southeast_lat_bottom = 30
        southeast_lat_top = 38
        
        southeast_lon_left  = 267
        southeast_lon_right = 282
    
    elif region_name == 'australia':
        if test_year == 2019:
            # start_ = '2019-06-01'
            # end_ = '2019-07-31'
    
            start_ = '2019-09-01'
            end_ = '2019-10-17'
    
    elif region_name == 'china':
        if test_year == 2019:
            start_ = '2019-07-14'
            end_ = '2019-08-28'
            # end_ = '2019-07-09
    return start_,end_

    