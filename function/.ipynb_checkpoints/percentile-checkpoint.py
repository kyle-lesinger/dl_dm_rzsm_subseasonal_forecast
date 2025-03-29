#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import os
from glob import glob
from typing import Tuple
import climpred
from climpred.options import OPTIONS
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import pickle
import dask

from . import masks
from . import conf

def datestdtojd (before_year, before_month, before_day):
    # convert dates to julian date to allow for window selection
    fmt='%Y-%m-%d %H:%M:%S'
    try:
        target_date = str(datetime(before_year, before_month, before_day))
    except NameError:
        target_date = str(dt.datetime(before_year, before_month, before_day))
    sdtdate = dt.datetime.strptime(target_date, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)

def jdtodatestd (jdate):
    #convert julian date to datetime 
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)

def create_data_julian_dates_GEFS(reforecast_file,region_name):
    dim_order = ['S','M','L','Y','X']
    
    new_save_dir = f'{conf.gefsv12_data}/{region_name}/soilw_bgrnd/RZSM_anomaly_with_julian_dates'
    os.system(f'mkdir -p {new_save_dir}')
    
    #Make the julian dates for easier processing
    for idx,date in enumerate(reforecast_file.S.values):
        # break
        date_out = f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'
        out_name = f'{new_save_dir}/soilw_bgrnd_{date_out}.nc'
        
        if os.path.exists(out_name):
            pass
        else:
            single_file = reforecast_file.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order)
            date_list = [single_file.S.values + np.timedelta64(i, 'D') for i in single_file.L.values]

            julian_dates = [datestdtojd(pd.to_datetime(i).year[0], pd.to_datetime(i).month[0], pd.to_datetime(i).day[0]) for i in date_list]
            single_file['L'] = julian_dates
            single_file.to_netcdf(out_name)

    print('Loading the julian day anomaly dataset')
    
    return(xr.open_mfdataset(f'{new_save_dir}/soil*',combine='nested',concat_dim=['S']))

def create_data_julian_dates_GEFS_testing_distribution(reforecast_file,region_name):
    dim_order = ['S','M','L','Y','X']
    
    new_save_dir = f'{conf.gefsv12_data}/{region_name}/soilw_bgrnd/RZSM_anomaly_with_julian_dates_testing_distribution'
    os.system(f'mkdir -p {new_save_dir}')
    
    #Make the julian dates for easier processing
    for idx,date in enumerate(reforecast_file.S.values):
        # break
        date_out = f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'
        out_name = f'{new_save_dir}/soilw_bgrnd_{date_out}.nc'
        
        if os.path.exists(out_name):
            pass
        else:
            single_file = reforecast_file.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order)
            date_list = [single_file.S.values + np.timedelta64(i, 'D') for i in single_file.L.values]

            julian_dates = [datestdtojd(pd.to_datetime(i).year[0], pd.to_datetime(i).month[0], pd.to_datetime(i).day[0]) for i in date_list]
            single_file['L'] = julian_dates
            single_file.to_netcdf(out_name)

    print('Loading the julian day anomaly dataset')
    
    return(xr.open_mfdataset(f'{new_save_dir}/soil*',combine='nested',concat_dim=['S']))

def create_data_julian_dates_UNET_experiment(reforecast_file, experiment_number,region_name):
    dim_order = ['S','M','L','Y','X']
    
    new_save_dir = f'Data/UNET_RZSM_anomaly_with_julian_dates/{region_name}'
    os.system(f'mkdir -p {new_save_dir}')
    
    #Make the julian dates for easier processing
    for idx,date in enumerate(reforecast_file.S.values):
        # break
        date_out = f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'
        out_name = f'{new_save_dir}/{experiment_number}_{region_name}_{date_out}.nc'
        
        if os.path.exists(out_name):
            pass
        else:
            single_file = reforecast_file.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order)
            date_list = [single_file.S.values + np.timedelta64(i, 'D') for i in single_file.L.values]

            julian_dates = [datestdtojd(pd.to_datetime(i).year[0], pd.to_datetime(i).month[0], pd.to_datetime(i).day[0]) for i in date_list]
            single_file['L'] = julian_dates
            single_file.to_netcdf(out_name)

    print('Loading the julian day anomaly dataset')
    
    return(xr.open_mfdataset(f'{new_save_dir}/{experiment_number}*',combine='nested',concat_dim=['S']))

def create_data_julian_dates_UNET_experiment_testing_distribution(reforecast_file, experiment_number,region_name, forecast_source):
    dim_order = ['S','M','L','Y','X']
    
    new_save_dir = f'Data/UNET_RZSM_anomaly_with_julian_dates_testing_distribution/{region_name}'
    os.system(f'mkdir -p {new_save_dir}')
    
    #Make the julian dates for easier processing
    for idx,date in enumerate(reforecast_file.S.values):
        # break
        date_out = f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'
        out_name = f'{new_save_dir}/{experiment_number}_{forecast_source}_{date_out}.nc'
        
        if os.path.exists(out_name):
            pass
        else:
            single_file = reforecast_file.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order)
            date_list = [single_file.S.values + np.timedelta64(i, 'D') for i in single_file.L.values]

            julian_dates = [datestdtojd(pd.to_datetime(i).year[0], pd.to_datetime(i).month[0], pd.to_datetime(i).day[0]) for i in date_list]
            single_file['L'] = julian_dates
            single_file.to_netcdf(out_name)

    print('Loading the julian day anomaly dataset')
    
    return(xr.open_mfdataset(f'{new_save_dir}/{experiment_number}*{forecast_source}*',combine='nested',concat_dim=['S']))

def create_data_julian_dates_ECMWF(reforecast_file):
    dim_order = ['S','M','L','Y','X']
    
    new_save_dir = f'Data/ECMWF/soilw_bgrnd_processed/CONUS/RZSM_anomaly_with_julian_dates'
    os.system(f'mkdir -p {new_save_dir}')
    
    #Make the julian dates for easier processing
    for idx,date in enumerate(reforecast_file.S.values):
        # break
        date_out = f'{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}'
        out_name = f'{new_save_dir}/soilw_bgrnd_{date_out}.nc'
        
        if os.path.exists(out_name):
            pass
        else:
            single_file = reforecast_file.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order)
            date_list = [single_file.S.values + np.timedelta64(i, 'D') for i in single_file.L.values]

            julian_dates = [datestdtojd(pd.to_datetime(i).year[0], pd.to_datetime(i).month[0], pd.to_datetime(i).day[0]) for i in date_list]
            single_file['L'] = julian_dates
            single_file.to_netcdf(out_name)

    print('Loading the julian day anomaly dataset')
    
    return(xr.open_mfdataset(f'{new_save_dir}/soil*',combine='nested',concat_dim=['S']))

def get_non_completed_days(final_dates,date_index,day_to_grab, anomaly_file, RZSM_name, save_dir,source):
    selected_days = final_dates[date_index:date_index + day_to_grab]
    dates_init = selected_days[0].split('-')
    month_init = int(dates_init[0])
    day_init = int(dates_init[1])
    
    non_completed_dates = []
    for j in selected_days:
        dates = j.split('-')
        month =  int(dates[0])
        day = int(dates[1])            
        #Now loop through each of the month_days and create the distribution
        run_dates = anomaly_file.sel(S=(anomaly_file['S.month'] == month) & (anomaly_file['S.day'] == day ))
        #Now check if files exists (don't run otherwise)
        saved_dates1 = run_dates.S.values
        saved_dates2 = [f'{RZSM_name}_{source}_{pd.to_datetime(i).year}-{pd.to_datetime(i).month:02}-{pd.to_datetime(i).day:02}.nc' for i in saved_dates1]

        completed_or_not = []
        for i in saved_dates2:
            if os.path.exists(f'{save_dir}/{i}'):
                completed_or_not.append(True)
            else:
                completed_or_not.append(False)
        if len(completed_or_not) == sum(completed_or_not):
            #All files are already completed
            pass
        else:
            non_completed_dates.append(j)
    return(non_completed_dates)
            
    