#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import os
from glob import glob
from typing import Tuple
from climpred.options import OPTIONS
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import pickle, dask, climpred

#Personal functions
from . import masks
from . import conf


def create_new_X_coord_values(file):
    '''Change longitude from 0-360 to a -180 to 0 and - to +180 format'''
    new_lon_values = [i+360 if i < 0 else i for i in file.X.values]
    return new_lon_values

def make_daily_lags() -> list:
    daily_lags = [i for i in np.arange(-84,0,7)] #append daily lags (which have maximum at 12 weeks)
    daily_lags.append(-1) #append for day 1 lag
    return(daily_lags)

def restrict_to_bounding_box(file: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    try:
        #This is for observations
        file = file.sel(latitude=slice(mask.Y.values[0],mask.Y.values[-1])).sel(longitude=slice(mask.X.values[0],mask.X.values[-1]))
    except KeyError:
        #This is for reforecast
        file = file.sel(Y=slice(mask.Y.values[0],mask.Y.values[-1])).sel(X=slice(mask.X.values[0],mask.X.values[-1]))
    except AttributeError:
        file = file.sel(latitude=slice(mask.latitude.values[0],mask.latitude.values[-1])).sel(longitude=slice(mask.longitude.values[0],mask.longitude.values[-1]))
    except KeyError:
        file = file.sel(Y=slice(mask.latitude.values[0],mask.latitude.values[-1])).sel(X=slice(mask.longitude.values[0],mask.longitude.values[-1]))
    
    return(file)

def xarray_varname(file: xr.DataArray) -> str:
    #grabs the first xarray variable, just easier with a function. But only works if you have 1 varaible in the xarray file
    return(list(file.keys())[0])


def _count_np_nan_values(file: xr.DataArray,var: str) -> None:
    arr_ = np.array(file[xarray_varname(file)].values)
    return(print(f'Number of np.nan values before rolling mean for {var}: {np.count_nonzero(np.isnan(arr_))}'))


def _count_zero_values(file: xr.DataArray,var: str) -> None:
    arr_ = np.array(file[xarray_varname(file)].values)
    return(print(f'Number of 0 values for {var}: {np.count_nonzero(arr_==0)}'))

def return_proper_mask_for_bounding(region_name: str) -> xr.DataArray:

    mask = masks.load_mask(region_name)
    try:
        print(f'\nLatitude values for mask is {mask.Y.values}')
        print(f'\nLongitude values for mask is {mask.X.values}')
    except AttributeError:
        print(f'\nLatitude values for mask is {mask.latitude.values}')
        print(f'\nLongitude values for mask is {mask.longitude.values}')

    return(mask)




def open_reanalysis_files_and_preprocess_rolling_mean_RZSM_only(path_to_file: str, file_variable: str, start_date: str, end_date: str, region_name: str) -> Tuple[xr.DataArray, list]:
    #Open file, apply 7-day rolling mean, restrict to specific bounding box
    mask = return_proper_mask_for_bounding(region_name)
    print('Loaded mask')
    file =  restrict_to_bounding_box(xr.open_dataset(path_to_file).sel(time=slice(start_date,end_date)), mask)
    print('Restricted to bounding box')
    _count_np_nan_values(file,file_variable)
    #apply 7-day rolling mean
    try:
        file = file.drop_vars('time_bnds')
        file = file.drop_vars('utc_date')
    except ValueError:
        print('Nothing to drop')
        
    file = file.rolling(time=7, min_periods=7,center=False).mean()
    proper_date_time = file.time.values
    return(file,proper_date_time)


def open_reanalysis_files_and_preprocess_rolling_mean_other_files(path_to_file: str, file_variable: str, start_date: str, end_date: str, region_name: str, proper_date_time: str) -> xr.DataArray:
    #Open file, apply 7-day rolling mean, restrict to CONUS bounding box
    #Use the proper date time from the very first processed observation. Just to make sure that they are all the exact same
    mask = return_proper_mask_for_bounding(region_name)

    try:
        file =  restrict_to_bounding_box(xr.open_dataset(path_to_file).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
    except ValueError:
        file =  restrict_to_bounding_box(xr.open_dataset(path_to_file).sel(time=slice(start_date,end_date)), mask)
    
    _count_np_nan_values(file,file_variable)
    
    file['time'] = proper_date_time
    #apply 7-day rolling mean
    file = file.rolling(time=7, min_periods=7,center=False).mean()
    return(file)


def open_temperature_files_and_preprocess_rolling_mean(path_to_tmax: str, path_to_tmin: str, file_variable_tmax: str, file_variable_tmin: str, start_date: str, end_date: str, region_name: str, proper_date_time: list) -> Tuple[xr.DataArray, xr.DataArray]:
    #Open tmx and tmin, change time coordinates, calculate difference between tmax and tmin, apply rolling mean, restrict to CONUS bounding box
    print('Creating tmax and difference in tmax and tmin variables.')
    mask = return_proper_mask_for_bounding(region_name)

    if region_name == 'CONUS':
        try:
            tmax =  restrict_to_bounding_box(xr.open_dataset(path_to_tmax).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            tmax =  restrict_to_bounding_box(xr.open_dataset(path_to_tmax).sel(forecast_initial_time=slice(start_date,end_date)), mask).rename({'forecast_initial_time':'time'})
    else:
        try:
            tmax =  restrict_to_bounding_box(xr.open_dataset(path_to_tmax).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            tmax =  restrict_to_bounding_box(xr.open_dataset(path_to_tmax).sel(time=slice(start_date,end_date)), mask)
    _count_np_nan_values(tmax,file_variable_tmax)

    tmax['time'] = proper_date_time

    if region_name == 'CONUS':
        try:
            tmin =  restrict_to_bounding_box(xr.open_dataset(path_to_tmin).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            tmin =  restrict_to_bounding_box(xr.open_dataset(path_to_tmin).sel(forecast_initial_time=slice(start_date,end_date)), mask).rename({'forecast_initial_time':'time'})
    else:
        try:
            tmin =  restrict_to_bounding_box(xr.open_dataset(path_to_tmin).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            tmin =  restrict_to_bounding_box(xr.open_dataset(path_to_tmin).sel(time=slice(start_date,end_date)), mask)
    _count_np_nan_values(tmin,file_variable_tmin)

    tmin['time'] = proper_date_time

    
    diff_tmax_tmin = np.subtract(tmax[xarray_varname(tmax)],tmin[xarray_varname(tmin)]).to_dataset() #calculate difference between tmax and tmin
    diff_tmax_tmin = diff_tmax_tmin.rename({xarray_varname(diff_tmax_tmin):'diff_tmax_tmin'})
    
    #apply rolling mean
    tmax = tmax.rolling(time=7, min_periods=7,center=False).mean()
    diff_tmax_tmin = diff_tmax_tmin.rolling(time=7, min_periods=7,center=False).mean().sel(time=slice(start_date,end_date))
    
    return(tmax,diff_tmax_tmin)

def calculate_2m_specific_humidity_and_preprocess(path_to_dewpoint: str, path_to_pressure: str, file_variable_dewpoint: str, file_variable_pressure: str, 
                                                  start_date: str, end_date: str, region_name: str, proper_date_time: list, obs_dir: str, spfh_unit: str) -> xr.DataArray:
    #Open surface pressure and 2m dewpoint temperature, change time coordinates, calculate surface specific humidity, apply rolling mean, restrict to CONUS bounding box
     
    save_spfh_ERA = f'{obs_dir}/surface_specific_humidity_merged.nc4'
    
    if os.path.exists(save_spfh_ERA):
        spfh_ERA = xr.open_dataset(save_spfh_ERA)
        print(f'\nLoading the previously created specific humidity from {obs_dir}')
    else:
        print('Creating specific humidity from surface pressure and 2m dewpoint.')
        mask = return_proper_mask_for_bounding(region_name)

        try:
            pressure =  restrict_to_bounding_box(xr.open_dataset(path_to_pressure).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            pressure =  restrict_to_bounding_box(xr.open_dataset(path_to_pressure).sel(time=slice(start_date,end_date)), mask)
            
        _count_np_nan_values(pressure,file_variable_pressure)
        pressure['time'] = proper_date_time

        try:
            dewpoint =  restrict_to_bounding_box(xr.open_dataset(path_to_dewpoint).sel(time=slice(start_date,end_date)), mask).drop('time_bnds')
        except ValueError:
            dewpoint =  restrict_to_bounding_box(xr.open_dataset(path_to_dewpoint).sel(time=slice(start_date,end_date)), mask)
            
        _count_np_nan_values(dewpoint,file_variable_dewpoint)
        dewpoint['time'] = proper_date_time

        #Create surface specific humidity (source: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.specific_humidity_from_dewpoint.html)
        
        spfh_ERA = dewpoint.copy(deep=True).rename({xarray_varname(dewpoint):'surface_spfh'})
        spfh_val = specific_humidity_from_dewpoint(pressure[xarray_varname(pressure)].values *units.Pa, dewpoint[xarray_varname(dewpoint)].values *units.K).to(spfh_unit)
        spfh_ERA.surface_spfh[:,:,:] = spfh_val
        _count_np_nan_values(spfh_ERA,'specific_humidity_obs')
        spfh_ERA = spfh_ERA.rolling(time=7, min_periods=7,center=False).mean()
    
        spfh_ERA.to_netcdf(f'{save_spfh_ERA}')
    
    return(spfh_ERA)

def return_GEFSv12_reforecast_files(dir_path: str, name_of_var: str, region_name: str) -> xr.DataArray:
    #Loads the reforecast file
    mask = return_proper_mask_for_bounding(region_name)
    
    print(f'\nLoading files for variable {name_of_var}')

    # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4', parallel = True, chunks={'X': 10, 'Y': 10}) #chunking & parallelizing doesn't seem to speed up anything

    try:
        # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4', parallel=True, chunks={'S': 'auto', 'X': 'auto', 'Y': 'auto', 'M': 'auto', 'L': 'auto'})
        # file = dask.array.from_netcdf(f'{dir_path}/{name_of_var}/*.nc4', chunks={'S': 10})
        # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4', chunks={'S':10})
        # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4')
        file = xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4', combine='nested', concat_dim='S')
    except OSError:
        # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc4', parallel=True, chunks={'S': 'auto', 'X': 'auto', 'Y': 'auto', 'M': 'auto', 'L': 'auto'})
        # file = dask.array.from_netcdf(f'{dir_path}/{name_of_var}/*.nc', chunks={'S': 10})
        # file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc', chunks={'S':10})
        file =xr.open_mfdataset(f'{dir_path}/{name_of_var}/*.nc')

    # file = dask.array.to_dask_dataframe(file)

    #We have different coordinate systems. So we need to add 360 to each of the X coordinates if they are negative
    if region_name == 'CONUS':
        new_X_coords = [i+360 if i < 0 else i for i in file.X.values]
        file = file.assign_coords({'X':new_X_coords})

    _count_np_nan_values(file, f'{name_of_var}_GEFSv12')
    
    print('Apply a 7-day rolling mean')
    file = restrict_to_bounding_box(file,mask)
    file_mean = file.rolling(L=7, min_periods=7,center=False).mean()

    #Just replace the very first day values to ensure we can compute lead day 0
    file_mean[xarray_varname(file)][:,:,0,:,:] = file[xarray_varname(file)][:,:,0,:,:]

    print(f'\nLoaded files for {name_of_var}')
    
    return(file_mean)


def return_reforecast_files_by_concatenation(dir_path: str, name_of_var: str, region_name: str) -> xr.DataArray:
    #Loads the reforecast file
    mask = return_proper_mask_for_bounding(region_name)
    
    print(f'\nLoading files for variable {name_of_var}')

    file_paths = sorted(glob(f'{dir_path}/{name_of_var}/*.n*'))

    # Open individual xarray datasets from the files
    datasets = [xr.open_dataset(file) for file in file_paths]

    #We have different coordinate systems. So we need to add 360 to each of the X coordinates if they are negative
    if region_name == 'CONUS':
        print(f'We are changing the coordinates of CONUS to match similar format as GLEAM')
        new_X_coords = create_new_X_coord_values(datasets[0])
        #Add the new coordinates
        datasets = [file.assign_coords({'X':new_X_coords}) for file in datasets]
        datasets = [restrict_to_bounding_box(file,mask) for file in datasets]
        
    # Concatenate the datasets along a specific dimension
    print(f'\nConcatenating files for {name_of_var}. Takes about 10 minutes for CONUS.')
    file = xr.concat(datasets, dim='S', combine_attrs="override")
    
    # file = dask.delayed(xr.concat)(datasets, dim="S")

   # Close individual datasets to free up resources
    for dataset in datasets:
        dataset.close()

    # Some files have a value of zero which is skewing results. So we need to replace with the mean.
    print('Masking files if they have value of zero')
    file_masked = xr.where(file == 0, np.nan, file)
    #Now take the mean 
    file_masked_mean = file_masked.mean(dim='S')
    #Now replace the values where there was a zero
    file_out = xr.where(file==0, file_masked_mean, file)

    _count_np_nan_values(file_out, f'{name_of_var}_GEFSv12')
    _count_zero_values(file_out, f'{name_of_var}_GEFSv12')
    
    print(f'Loaded files for {name_of_var}')
    
    return(file_out)

def return_ECMWF_reforecast_files_by_concatenation(dir_path: str, name_of_var: str, region_name: str, init_date_list: list, template) -> xr.DataArray:
    #Loads the reforecast file
    mask = return_proper_mask_for_bounding(region_name)
    
    print(f'\nLoading files for variable {name_of_var}')

    file_paths = sorted(glob(f'{dir_path}/{name_of_var}*.n*'))
    good_dates = []
    for i in init_date_list:
        # break
        good_dates.append([j for j in file_paths if i in j])

    #NOw combined the lists
    combined_list = [item for sublist in good_dates for item in sublist]
    
    # Open individual xarray datasets from the files
    datasets = [xr.open_dataset(file) for file in combined_list]

    
    #We have different coordinate systems. So we need to add 360 to each of the X coordinates if they are negative
    if region_name == 'CONUS':
        print(f'We are changing the coordinates of CONUS to match similar format as GLEAM')
        new_X_coords = create_new_X_coord_values(template)
        #Add the new coordinates
        datasets = [file.assign_coords({'X':new_X_coords}) for file in datasets]
        datasets = [restrict_to_bounding_box(file,mask) for file in datasets]
        
    # Concatenate the datasets along a specific dimension
    print(f'\nConcatenating files for {name_of_var}. Takes about 10 minutes for {region_name}.')
    file = xr.concat(datasets, dim='S', combine_attrs="override")
    
    # file = dask.delayed(xr.concat)(datasets, dim="S")

   # Close individual datasets to free up resources
    for dataset in datasets:
        dataset.close()

    
    
    # Some files have a value of zero which is skewing results. So we need to replace with the mean.
    print('Masking files if they have value of zero')
    file_masked = xr.where(file == 0, np.nan, file)
    #Now take the mean 
    file_masked_mean = file_masked.mean(dim='S')
    #Now replace the values where there was a zero
    file_out = xr.where(file==0, file_masked_mean, file)
    
    _count_np_nan_values(file_out, f'{name_of_var}_ECMWF')
    _count_zero_values(file_out, f'{name_of_var}_ECMWF')
    
    print(f'Loaded files for {name_of_var}')
    
    return(file_out)

def get_init_date_list(forecast_variable_path: str) -> list:
    #returns the init date list from the forecast variable path. Used for creating observational data that is in the same format as the SubX data.
    file_list = sorted(glob(f'{forecast_variable_path}/*.n*'))
    date_list = [i.split('.')[0].split('_')[-1] for i in file_list]
    return date_list


def create_seasonal_anomaly(file: xr.DataArray,train_end: int) -> xr.DataArray:
    #First create a climatology by season based on training data only (mean is coming directly from training data and applied to validation and testing)

    print(f'\nLongitude values of file we are performing anomaly on are {file.X.values}')
    print(f'\nLatitude values of file we are performing anomaly on are {file.Y.values}')

    print(f'\nGetting all data before year {train_end+1}\n')
    
    climpred.set_options(seasonality="season") 
    seasonality_str = OPTIONS["seasonality"]
    
    #change the dates to be pd.to_datetime objects
    file['S'] = pd.to_datetime(file.S.values)

    # #If performing anomaly based on lead
    # out_file_by_season = file.copy(deep=True)

    # print('Making anomalies for each individual lead')
    # for idx,lead in enumerate(file.L.values):
    #     # break
    #     climatology_season = file.sel(S=(file['S.year'] <= train_end)).sel(L=lead).groupby(f"S.{seasonality_str}").mean()
        
    #     summer_= file.sel(S=(file['S.season']=='JJA')).sel(L=lead) - climatology_season.sel(season='JJA')
    #     fall_= file.sel(S=(file['S.season']=='SON')).sel(L=lead)- climatology_season.sel(season='SON')
    #     winter_= file.sel(S=(file['S.season']=='DJF')).sel(L=lead)- climatology_season.sel(season='DJF')
    #     spring_= file.sel(S=(file['S.season']=='MAM')).sel(L=lead)- climatology_season.sel(season='MAM')
    
    #     combined_files = xr.concat([summer_,fall_,winter_,spring_],dim='S').sortby('S')
    #     combined_files = combined_files.drop('season')
    #     out_file_by_season[xarray_varname(out_file_by_season)][:,:,idx,:,:] = combined_files[xarray_varname(out_file_by_season)][:,:,:,:]
    # return(out_file_by_season)


    print('Making anomalies for each individual lead')

    climatology_season = file.sel(S=(file['S.year'] <= train_end)).groupby(f"S.{seasonality_str}").mean()
    
    summer_= file.sel(S=(file['S.season']=='JJA')) - climatology_season.sel(season='JJA')
    fall_= file.sel(S=(file['S.season']=='SON'))- climatology_season.sel(season='SON')
    winter_= file.sel(S=(file['S.season']=='DJF'))- climatology_season.sel(season='DJF')
    spring_= file.sel(S=(file['S.season']=='MAM'))- climatology_season.sel(season='MAM')

    combined_files = xr.concat([summer_,fall_,winter_,spring_],dim='S').sortby('S')
    combined_files = combined_files.drop('season')
    
    return(combined_files, climatology_season) #combine all anomalies, sort by date


def create_seasonal_anomaly_with_different_testing_years(file: xr.DataArray,train_start, train_end, train_start2, train_end2) -> xr.DataArray:
    #First create a climatology by season based on training data only (mean is coming directly from training data and applied to validation and testing)

    print(f'\nLongitude values of file we are performing anomaly on are {file.X.values}')
    print(f'\nLatitude values of file we are performing anomaly on are {file.Y.values}')

    print(f'\nGetting all data before year {train_end+1}\n')
    
    climpred.set_options(seasonality="season") 
    seasonality_str = OPTIONS["seasonality"]
    
    #change the dates to be pd.to_datetime objects
    file['S'] = pd.to_datetime(file.S.values)

    # #If performing anomaly based on lead
    # out_file_by_season = file.copy(deep=True)

    # print('Making anomalies for each individual lead')
    # for idx,lead in enumerate(file.L.values):
    #     # break
    #     climatology_season = file.sel(S=(file['S.year'] <= train_end)).sel(L=lead).groupby(f"S.{seasonality_str}").mean()
        
    #     summer_= file.sel(S=(file['S.season']=='JJA')).sel(L=lead) - climatology_season.sel(season='JJA')
    #     fall_= file.sel(S=(file['S.season']=='SON')).sel(L=lead)- climatology_season.sel(season='SON')
    #     winter_= file.sel(S=(file['S.season']=='DJF')).sel(L=lead)- climatology_season.sel(season='DJF')
    #     spring_= file.sel(S=(file['S.season']=='MAM')).sel(L=lead)- climatology_season.sel(season='MAM')
    
    #     combined_files = xr.concat([summer_,fall_,winter_,spring_],dim='S').sortby('S')
    #     combined_files = combined_files.drop('season')
    #     out_file_by_season[xarray_varname(out_file_by_season)][:,:,idx,:,:] = combined_files[xarray_varname(out_file_by_season)][:,:,:,:]
    # return(out_file_by_season)


    print('Making anomalies for each individual lead')

    train_subset1 = file.sel(S=(file['S.year'] <= train_end))
    train_subset2 = file.sel(S=((file['S.year'] >= train_start2) & (file['S.year'] <= train_end2)))

    #Now combine 
    ct = xr.concat([train_subset1, train_subset2],dim='S')
    climatology_season = ct.groupby(f"S.{seasonality_str}").mean()
    
    summer_= file.sel(S=(file['S.season']=='JJA')) - climatology_season.sel(season='JJA')
    fall_= file.sel(S=(file['S.season']=='SON'))- climatology_season.sel(season='SON')
    winter_= file.sel(S=(file['S.season']=='DJF'))- climatology_season.sel(season='DJF')
    spring_= file.sel(S=(file['S.season']=='MAM'))- climatology_season.sel(season='MAM')

    combined_files = xr.concat([summer_,fall_,winter_,spring_],dim='S').sortby('S')
    combined_files = combined_files.drop('season')
    
    return(combined_files, climatology_season) #combine all anomalies, sort by date



def plot_distribution_by_lead_datashader(data_file: xr.DataArray, name_of_var_and_source: str, obs_or_forecast: str, anomaly_or_min_max: str, region_name: str, lead_select: list) -> None:
    #Compares the reforecast values with the actual observation percentile values
    
    save_dir_plot = f'Outputs/{anomaly_or_min_max}_distribution_plots/{region_name}'
    os.system(f'mkdir -p {save_dir_plot}')
    
    def remove_np_nan(file):
        return(file[~np.isnan(file)])
    if obs_or_forecast == 'obs':
        #Plot only a single model because they are repated across all models

        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).sel(M=7).values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).sel(M=7).values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).sel(M=7).values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).sel(M=7).values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).sel(M=7).values.flatten()
    else:
        print('Plotting all models')
        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).mean(dim='M').values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).mean(dim='M').values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).mean(dim='M').values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).mean(dim='M').values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).mean(dim='M').values.flatten()
    # flat_data6 = data_file[xarray_varname(data_file)].sel(L=leads_[5]).sel(M=7).values.flatten()
    
    data_arrays = [flat_data1,flat_data2,flat_data3,flat_data4,flat_data5]

    
    fig,axs = plt.subplots(1,len(lead_select),figsize=(16,5))
    axs.flatten()
    
    idx_start = 0
    for idx,arr in enumerate(data_arrays):
        sns.kdeplot(remove_np_nan(arr), ax=axs[idx_start])
        axs[idx_start].set_title(f'Lead Day {lead_select[idx]}', fontsize=10)
        idx_start +=1
        
    plt.suptitle(f'{name_of_var_and_source} {anomaly_or_min_max} distribution',fontsize=20)
    plt.savefig(f'{save_dir_plot}/{name_of_var_and_source}_distribution_after_{anomaly_or_min_max}_calc.png')

    return(0)

def plot_distribution_by_lead_datashader_with_different_testing_years(data_file: xr.DataArray, name_of_var_and_source: str, obs_or_forecast: str, anomaly_or_min_max: str, region_name: str, lead_select: list, test_start: int, test_end: int) -> None:
    #Compares the reforecast values with the actual observation percentile values
    
    save_dir_plot = f'Outputs/{anomaly_or_min_max}_distribution_plots/{region_name}'
    os.system(f'mkdir -p {save_dir_plot}')
    
    def remove_np_nan(file):
        return(file[~np.isnan(file)])
    if obs_or_forecast == 'obs':
        #Plot only a single model because they are repated across all models

        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).sel(M=7).values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).sel(M=7).values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).sel(M=7).values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).sel(M=7).values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).sel(M=7).values.flatten()
    else:
        print('Plotting all models')
        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).mean(dim='M').values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).mean(dim='M').values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).mean(dim='M').values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).mean(dim='M').values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).mean(dim='M').values.flatten()
    # flat_data6 = data_file[xarray_varname(data_file)].sel(L=leads_[5]).sel(M=7).values.flatten()
    
    data_arrays = [flat_data1,flat_data2,flat_data3,flat_data4,flat_data5]

    
    fig,axs = plt.subplots(1,len(lead_select),figsize=(16,5))
    axs.flatten()
    
    idx_start = 0
    for idx,arr in enumerate(data_arrays):
        sns.kdeplot(remove_np_nan(arr), ax=axs[idx_start])
        axs[idx_start].set_title(f'Lead Day {lead_select[idx]}', fontsize=10)
        idx_start +=1
        
    plt.suptitle(f'{name_of_var_and_source} {anomaly_or_min_max} distribution',fontsize=20)
    plt.savefig(f'{save_dir_plot}/{name_of_var_and_source}_distribution_after_{anomaly_or_min_max}_calc_test_years_{test_start}-{test_end}.png')

    return(0)

def plot_distribution_by_lead_datashader_by_individual_grid_cell(data_file: xr.DataArray, name_of_var_and_source: str, obs_or_forecast: str, anomaly_or_min_max: str, region_name: str, lead_select: list) -> None:
    #Compares the reforecast values with the actual observation percentile values
    
    save_dir_plot = f'Outputs/{anomaly_or_min_max}_distribution_plots/{region_name}'
    os.system(f'mkdir -p {save_dir_plot}')
    
    def remove_np_nan(file):
        return(file[~np.isnan(file)])
    if obs_or_forecast == 'obs':
        #Plot only a single model because they are repated across all models

        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).sel(M=7).values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).sel(M=7).values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).sel(M=7).values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).sel(M=7).values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).sel(M=7).values.flatten()
    else:
        print('Plotting all models')
        flat_data1 = data_file[xarray_varname(data_file)].sel(L=lead_select[0]).mean(dim='M').values.flatten()
        flat_data2 = data_file[xarray_varname(data_file)].sel(L=lead_select[1]).mean(dim='M').values.flatten()
        flat_data3 = data_file[xarray_varname(data_file)].sel(L=lead_select[2]).mean(dim='M').values.flatten()
        flat_data4 = data_file[xarray_varname(data_file)].sel(L=lead_select[3]).mean(dim='M').values.flatten()
        flat_data5 = data_file[xarray_varname(data_file)].sel(L=lead_select[4]).mean(dim='M').values.flatten()
    # flat_data6 = data_file[xarray_varname(data_file)].sel(L=leads_[5]).sel(M=7).values.flatten()
    
    data_arrays = [flat_data1,flat_data2,flat_data3,flat_data4,flat_data5]

    
    fig,axs = plt.subplots(1,len(lead_select),figsize=(16,5))
    axs.flatten()
    
    idx_start = 0
    for idx,arr in enumerate(data_arrays):
        sns.kdeplot(remove_np_nan(arr), ax=axs[idx_start])
        axs[idx_start].set_title(f'Lead Day {lead_select[idx]}', fontsize=10)
        idx_start +=1
        
    plt.suptitle(f'{name_of_var_and_source} {anomaly_or_min_max} distribution',fontsize=20)
    plt.savefig(f'{save_dir_plot}/{name_of_var_and_source}_distribution_after_{anomaly_or_min_max}_by_grid_cell_calc.png')

    return(0)


#Now save the RZSM anomalies from reforecast
def save_baseline_RZSM_anomaly(anom: xr.DataArray, region_name: str, fcst_dir: str, ) -> None:
    anom = anom.load()

    save_dir = f'{fcst_dir}/baseline_RZSM_anomaly'
    os.system(f'mkdir -p {save_dir}')
        

    dim_order = ['S','M','L','Y','X']

    for date in anom.S.values:
        # break
        filename=f'{save_dir}/soilw_bgrnd_{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}.nc'
        anom.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order).astype(np.float32).to_netcdf(filename)

    return('Completed saving anomaly baseline values')

#Now save the RZSM anomalies from reforecast
def save_GEFS_min_max(anom: xr.DataArray, region_name: str, fcst_dir: str) -> None:
    anom = anom.load()

    if (region_name == 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = 'Data/GEFSv12_reforecast/soilw_bgrnd/min_max_RZSM'
    elif (region_name != 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/min_max_RZSM'
    elif ('ECMWF' in fcst_dir):
        save_dir = f'{fcst_dir}/min_max_RZSM'
        
    os.system(f'mkdir -p {save_dir}')

    dim_order = ['S','M','L','Y','X']

    for date in anom.S.values:
        # break
        filename=f'{save_dir}/soilw_bgrnd_{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}.nc'
        if os.path.exists(filename):
            pass
        else:
            anom.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order).astype(np.float32).to_netcdf(filename)

    return('Completed saving min max values')

#Now save the RZSM anomalies from reforecast
def save_residuals(anom: xr.DataArray, region_name: str, fcst_dir: str) -> None:
    anom = anom.load()

    if (region_name == 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = 'Data/GEFSv12_reforecast/soilw_bgrnd/anomaly_residuals_RZSM'
    elif (region_name != 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/anomaly_residuals_RZSM'
    elif ('ECMWF' in fcst_dir):
        save_dir = f'{fcst_dir}/min_max_RZSM'
        
    os.system(f'mkdir -p {save_dir}')

    dim_order = ['S','M','L','Y','X']

    for date in anom.S.values:
        # break
        filename=f'{save_dir}/soilw_bgrnd_{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}.nc'
        if os.path.exists(filename) and  os.path.exists('this.nc'):
            pass
        else:
            anom.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order).astype(np.float32).to_netcdf(filename)

    return('Completed saving anomaly residual in values into {save_dir}.')

#Now save the RZSM anomalies from reforecast
def save_residuals_min_max_RZSM_OBS_minus_GEFS(anom: xr.DataArray, region_name: str, fcst_dir: str) -> None:
    anom = anom.load()

    if (region_name == 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = 'Data/GEFSv12_reforecast/soilw_bgrnd/residuals_RZSM_min_max'
    elif (region_name != 'CONUS') and ('GEFSv12' in fcst_dir):
        save_dir = f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/residuals_RZSM_min_max'
    elif ('ECMWF' in fcst_dir):
        save_dir = f'{fcst_dir}/residuals_RZSM_min_max'
        
    os.system(f'mkdir -p {save_dir}')

    dim_order = ['S','M','L','Y','X']

    for date in anom.S.values:
        # break
        filename=f'{save_dir}/soilw_bgrnd_{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}.nc'
        if os.path.exists(filename):
            pass
        else:
            anom.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order).astype(np.float32).to_netcdf(filename)

    return('Completed saving residuals min max values')

#Now save the RZSM anomalies from reforecast
def save_baseline_ECMWF_RZSM_anomaly(anom: xr.DataArray, region_name: str) -> None:
    anom = anom.load()

    save_dir = f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly'

    os.system(f'mkdir -p {save_dir}')

    dim_order = ['S','M','L','Y','X']

    for date in anom.S.values:
        # break
        filename=f'{save_dir}/soilw_bgrnd_{pd.to_datetime(date).year}-{pd.to_datetime(date).month:02}-{pd.to_datetime(date).day:02}.nc'
        print(f'Saving as {filename}')
        
        if os.path.exists(filename):
            pass
        else:
            anom.sel(S=date).expand_dims({'S': 1}).transpose(*dim_order).to_netcdf(filename)

    return('Completed saving anomaly baseline values')




def choose_training_years_and_min_max_scale(file: xr.DataArray, train_end: int,variable: str, obs_or_forecast: str, region_name: str, obs_min_max: xr.DataArray, lead_select: list ) -> xr.DataArray:
    # For nomalization we want to subtract the mean and divide by standard deviation.
    # We are looking at the mean and standard deviation of all ensemble members
    # for the entire time series (not individual grid cells). And afterward we are 
    # converting the np.nan values to zero for RZSM only
    
    save_min_max_dir = f'Data/min_max_values/{region_name}'
    os.system(f'mkdir -p {save_min_max_dir}')

    daily_lags = make_daily_lags()
    #We are not loading into memory at this point
    if obs_or_forecast == 'obs':
        leads_ = lead_select
        leads_ = daily_lags+leads_
    else:
        leads_ = lead_select

    #Subset based on leads
    file = file.sel(L=leads_).load()

    print(f'\nWorking on lead {file.L.values} for min max scaling... This is the size of the input')
    
    '''First, make sure to change all values with 0 to np.nan (really only an issue with RZSM_GEFSv12)'''
    if ('soil' in variable) or ('residual' in variable):
        if 'residual' in variable:
            pass
            
        elif 'GEFSv12' in variable:
            #Make sure that we have the same values of np.nan in the same places as the observation files
            #This really only applies to a few grid cells but we are trying to be more precise
            
            #Just grab a single lead from the OBS file file to help with masking
            file = file.where(obs_min_max[xarray_varname(obs_min_max)].isel(L=3) != 0,np.nan)
        
        else:
            file = file.where(file[xarray_varname(file)] != 0,np.nan)
    
    #Get the training years 
    train=file.sel(S=file['S.year'] <= train_end)
    
    print(f'\nCreating min max for {variable}. This takes a while because it is not loaded into memory.')
    print(f'\nTraining file dates with which we are creating the min and maximum files for scaling: {train.S.values[0]}')
    print(f'\nTraining file dates end which we are creating the min and maximum files for scaling: {train.S.values[-1]}')
    print(f'\nTraining lead values: {train.L.values}')
    #find min and max (faster computation with xarray function than numpy function)

    out_file = file.copy(deep=True)
    for idx,lead in enumerate(leads_):
            
        save_max = f'{save_min_max_dir}/{variable}_lead{lead}_max.npy'
        save_min = f'{save_min_max_dir}/{variable}_lead{lead}_min.npy'

        max_, min_ = train.sel(L=lead).max().compute(), train.sel(L=lead).min().compute()
        max_ = max_.to_array().values[0]
        np.save(save_max,max_)
        
        print(f'\nMaximum value for variable {variable} is {max_} for lead {lead}.')
        
        min_ = min_.to_array().values[0]
        np.save(save_min,min_)
    
        print(f'\nMinimum value for variable {variable} is {min_} for lead {lead}.')
        
        max_min = max_ - min_
    
        print(f'\nMaximum minus minimum value for variable {variable} is {max_min}.')
        
        print(f'\nStandardizing data with min max for {variable}. (x-min) / (max - min)') 
        #perform min max scaling and return
    
        out_file[xarray_varname(out_file)][:,:,idx,:,:] = np.divide(np.subtract(file[xarray_varname(file)].sel(L=lead),min_),max_min)

    #Fill RZSM np.nan values with 0. Also this might catch some other values which I don't know if they are zero.
    out_file = out_file.fillna(0)

    _count_np_nan_values(out_file, variable)
    _count_zero_values(out_file, variable)
    
    #Just for verification (our data looks good)
    # count_nan = scaled_data.isnull().sum().to_array().values[0]
    # print(f'Number of np.nan values is {count_nan}')
    
    return out_file


def choose_training_years_and_min_max_scale_with_different_testing_years(file: xr.DataArray, train_start: int, train_end: int, train_start2: int, train_end2: int,
                                                                         val_start: int, val_end:int, test_start:int, test_end: int,
                                                                         variable: str, obs_or_forecast: str, region_name: str, 
                                                                         obs_min_max: xr.DataArray, lead_select: list ) -> xr.DataArray:
    # For nomalization we want to subtract the mean and divide by standard deviation.
    # We are looking at the mean and standard deviation of all ensemble members
    # for the entire time series (not individual grid cells). And afterward we are 
    # converting the np.nan values to zero for RZSM only
    
    save_min_max_dir = f'Data/min_max_values/{region_name}'
    os.system(f'mkdir -p {save_min_max_dir}')

    daily_lags = make_daily_lags()
    #We are not loading into memory at this point
    if obs_or_forecast == 'obs':
        leads_ = lead_select
        leads_ = daily_lags+leads_
    else:
        leads_ = lead_select

    #Subset based on leads
    file = file.sel(L=leads_).load()

    print(f'\nWorking on lead {file.L.values} for min max scaling... This is the size of the input')
    
    '''First, make sure to change all values with 0 to np.nan (really only an issue with RZSM_GEFSv12)'''
    if 'soil' in variable:
        if 'GEFSv12' in variable:
            #Make sure that we have the same values of np.nan in the same places as the observation files
            #This really only applies to a few grid cells but we are trying to be more precise
            
            #Just grab a single lead from the OBS file file to help with masking
            file = file.where(obs_min_max[xarray_varname(obs_min_max)].isel(L=3) != 0,np.nan)
        else:
            file = file.where(file[xarray_varname(file)] != 0,np.nan)
    
    #Get the training years 
    print('Making anomalies for each individual lead')
    train_subset1 = file.sel(S=(file['S.year'] <= train_end))
    train_subset2 = file.sel(S=((file['S.year'] >= train_start2) & (file['S.year'] <= train_end2)))

    #Now combine 
    train = xr.concat([train_subset1, train_subset2],dim='S').sortby('S')

    
    print(f'\nCreating min max for {variable}. This takes a while because it is not loaded into memory.')
    print(f'\nTraining file dates with which we are creating the min and maximum files for scaling: {train.S.values[0]}')
    print('\n We have a split training set for these dates.')
    print(f'\nTraining file dates end which we are creating the min and maximum files for scaling: {train.S.values[-1]}')
    print(f'\nTraining lead values: {train.L.values}')
    #find min and max (faster computation with xarray function than numpy function)

    out_file = file.copy(deep=True)
    for idx,lead in enumerate(leads_):
            
        save_max = f'{save_min_max_dir}/{variable}_lead{lead}_max_{test_end}.npy'
        save_min = f'{save_min_max_dir}/{variable}_lead{lead}_min_{test_end}.npy'

        max_, min_ = train.sel(L=lead).max().compute(), train.sel(L=lead).min().compute()
        max_ = max_.to_array().values[0]
        np.save(save_max,max_)
        
        print(f'\nMaximum value for variable {variable} is {max_} for lead {lead}.')
        
        min_ = min_.to_array().values[0]
        np.save(save_min,min_)
    
        print(f'\nMinimum value for variable {variable} is {min_} for lead {lead}.')
        
        max_min = max_ - min_
    
        print(f'\nMaximum minus minimum value for variable {variable} is {max_min}.')
        
        print(f'\nStandardizing data with min max for {variable}. (x-min) / (max - min)') 
        #perform min max scaling and return
    
        out_file[xarray_varname(out_file)][:,:,idx,:,:] = np.divide(np.subtract(file[xarray_varname(file)].sel(L=lead),min_),max_min)

    #Fill RZSM np.nan values with 0. Also this might catch some other values which I don't know if they are zero.
    out_file = out_file.fillna(0)

    _count_np_nan_values(out_file, variable)
    _count_zero_values(out_file, variable)
    
    #Just for verification (our data looks good)
    # count_nan = scaled_data.isnull().sum().to_array().values[0]
    # print(f'Number of np.nan values is {count_nan}')
    
    return out_file


def choose_training_years_and_min_max_scale_mean(file: xr.DataArray, train_end: int,variable: str, obs_or_forecast: str, region_name: str, obs_min_max: xr.DataArray, lead_select: list ) -> xr.DataArray:
    # For nomalization we want to subtract the mean and divide by standard deviation.
    # We are looking at the mean and standard deviation of all ensemble members
    # for the entire time series (not individual grid cells). And afterward we are 
    # converting the np.nan values to zero for RZSM only
    
    save_min_max_dir = f'Data/min_max_values/{region_name}'
    os.system(f'mkdir -p {save_min_max_dir}')

    daily_lags = make_daily_lags()
    #We are not loading into memory at this point
    if obs_or_forecast == 'obs':
        leads_ = lead_select
        leads_ = daily_lags+leads_
    else:
        leads_ = lead_select

    #Subset based on leads
    file = file.sel(L=leads_).load()

    print(f'\nWorking on lead {file.L.values} for min max scaling... This is the size of the input')
    
    '''First, make sure to change all values with 0 to np.nan (really only an issue with RZSM_GEFSv12)'''
    if 'soil' in variable:
        if 'GEFSv12' in variable:
            #Make sure that we have the same values of np.nan in the same places as the observation files
            #This really only applies to a few grid cells but we are trying to be more precise
            
            #Just grab a single lead from the OBS file file to help with masking
            file = file.where(obs_min_max[xarray_varname(obs_min_max)].isel(L=3) != 0,np.nan)
        elif 'ECMWF' in variable:
            file = file.where(obs_min_max[xarray_varname(obs_min_max)].isel(L=3) == 0,np.nan)
        else:
            file = file.where(file[xarray_varname(file)] != 0,np.nan)
    
    #Get the training years 
    train=file.sel(S=file['S.year'] <= train_end)
    
    print(f'\nCreating min max for {variable}. This takes a while because it is not loaded into memory.')
    print(f'\nTraining file dates with which we are creating the min and maximum files for scaling: {train.S.values[0]}')
    print(f'\nTraining file dates end which we are creating the min and maximum files for scaling: {train.S.values[-1]}')
    print(f'\nTraining lead values: {train.L.values}')
    #find min and max (faster computation with xarray function than numpy function)

    out_file = file.copy(deep=True)
    for idx,lead in enumerate(leads_):
            
        save_max = f'{save_min_max_dir}/{variable}_lead{lead}_mean_max.npy'
        save_min = f'{save_min_max_dir}/{variable}_lead{lead}_mean_min.npy'

        max_, min_ = train.sel(L=lead).max().compute(), train.sel(L=lead).min().compute()
        max_ = max_.to_array().values[0]
        np.save(save_max,max_)
        
        print(f'\nMaximum value for variable {variable} is {max_} for lead {lead}.')
        
        min_ = min_.to_array().values[0]
        np.save(save_min,min_)
    
        print(f'\nMinimum value for variable {variable} is {min_} for lead {lead}.')
        
        max_min = max_ - min_
    
        print(f'\nMaximum minus minimum value for variable {variable} is {max_min}.')
        
        print(f'\nStandardizing data with min max for {variable}. (x-min) / (max - min)') 
        #perform min max scaling and return
    
        out_file[xarray_varname(out_file)][:,:,idx,:,:] = np.divide(np.subtract(file[xarray_varname(file)].sel(L=lead),min_),max_min)

    #Fill RZSM np.nan values with 0. Also this might catch some other values which I don't know if they are zero.
    out_file = out_file.fillna(0)

    _count_np_nan_values(out_file, variable)
    _count_zero_values(out_file, variable)
    
    #Just for verification (our data looks good)
    # count_nan = scaled_data.isnull().sum().to_array().values[0]
    # print(f'Number of np.nan values is {count_nan}')
    
    return out_file

def choose_training_years_and_min_max_scale_by_individual_grid_cell(file: xr.DataArray, train_end: int, variable: str, obs_or_forecast: str, region_name: str, obs_min_max: xr.DataArray, lead_select: list ) -> xr.DataArray:
    # For nomalization we want to subtract the mean and divide by standard deviation.
    # We are looking at the mean and standard deviation of all ensemble members
    # for the entire time series (not individual grid cells). And afterward we are 
    # converting the np.nan values to zero for RZSM only
    
    save_min_max_dir = f'Data/min_max_values/{region_name}'
    os.system(f'mkdir -p {save_min_max_dir}')

    daily_lags = make_daily_lags()
    #We are not loading into memory at this point
    if obs_or_forecast == 'obs':
        leads_ = lead_select
        leads_ = daily_lags+leads_
    else:
        leads_ = lead_select

    #Subset based on leads
    file = file.sel(L=leads_).load()

    print(f'\nWorking on lead {file.L.values} for min max scaling... This is the size of the input')
    
    '''First, make sure to change all values with 0 to np.nan (really only an issue with RZSM_GEFSv12)'''
    if 'soil' in variable:
        if 'GEFSv12' in variable:
            #Make sure that we have the same values of np.nan in the same places as the observation files
            #This really only applies to a few grid cells but we are trying to be more precise
            
            #Just grab a single lead from the OBS file file to help with masking
            file = file.where(obs_min_max[xarray_varname(obs_min_max)].isel(L=3) != 0,np.nan)
        else:
            file = file.where(file[xarray_varname(file)] != 0,np.nan)
    
    #Get the training years 
    train=file.sel(S=file['S.year'] <= train_end)
    
    print(f'\nCreating min max for {variable}. This takes a while because it is not loaded into memory.')
    print(f'\nTraining file dates with which we are creating the min and maximum files for scaling: {train.S.values[0]}')
    print(f'\nTraining file dates end which we are creating the min and maximum files for scaling: {train.S.values[-1]}')
    print(f'\nTraining lead values: {train.L.values}')
    #find min and max (faster computation with xarray function than numpy function)

    out_file = file.copy(deep=True)
    for idx,lead in enumerate(leads_):
            
        save_max = f'{save_min_max_dir}/{variable}_lead{lead}_by_grid_cell_max.npy'
        save_min = f'{save_min_max_dir}/{variable}_lead{lead}_by_grid_cell_min.npy'

        max_, min_ = train.sel(L=lead).max(dim=['S','M']).compute(), train.sel(L=lead).min(dim=['S','M']).compute()
        max_ = max_.to_array().values[0]
        np.save(save_max,max_)
        
        print(f'\nMaximum value for variable {variable} is {max_} for lead {lead}.')
        
        min_ = min_.to_array().values[0]
        np.save(save_min,min_)
    
        print(f'\nMinimum value for variable {variable} is {min_} for lead {lead}.')
        
        max_min = max_ - min_
    
        print(f'\nMaximum minus minimum value for variable {variable} is {max_min}.')
        
        print(f'\nStandardizing data with min max for {variable}. (x-min) / (max - min)') 
        #perform min max scaling and return
    
        out_file[xarray_varname(out_file)][:,:,idx,:,:] = np.divide(np.subtract(file[xarray_varname(file)].sel(L=lead),min_),max_min)

    #Fill RZSM np.nan values with 0. Also this might catch some other values which I don't know if they are zero.
    out_file = out_file.fillna(0)

    _count_np_nan_values(out_file, variable)
    _count_zero_values(out_file, variable)
    
    #Just for verification (our data looks good)
    # count_nan = scaled_data.isnull().sum().to_array().values[0]
    # print(f'Number of np.nan values is {count_nan}')
    
    return out_file

def create_stacked_files_by_lead_for_verification(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'Data/model_npy_input_data/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        train_set = file.sel(S=(file['S.year'] <= train_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        val_set = file.sel(S=(file['S.year'] > train_end))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        test_set = file.sel(S=(file['S.year'] >= test_start)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        
        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_masked.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_masked.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_masked.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)


def create_stacked_files_by_lead_for_verification_residual_min_max(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'Data/model_npy_input_data/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        train_set = file.sel(S=(file['S.year'] <= train_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        val_set = file.sel(S=(file['S.year'] > train_end))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        test_set = file.sel(S=(file['S.year'] >= test_start)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        
        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_masked.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_masked.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_masked.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)


def create_stacked_files_by_lead_for_verification_TIMESERIES_CLASS(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'timeseries/Data/model_npy_inputs/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        train_set = file.sel(S=(file['S.year'] <= train_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        val_set = file.sel(S=(file['S.year'] > train_end))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        test_set = file.sel(S=(file['S.year'] >= test_start)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        
        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_masked.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_masked.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_masked.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)


def create_stacked_files_by_lead_for_verification_with_different_testing_years(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list, train_start: int, train_start2: int, train_end2: int, val_start: int, test_end: int):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'Data/model_npy_input_data/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_start,train_end,train_start2, train_end2, val_start,val_end,test_start,test_end,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        #Get the training years 
        train_subset1 = file.sel(S=(file['S.year'] <= train_end))
        train_subset2 = file.sel(S=((file['S.year'] >= train_start2) & (file['S.year'] <= train_end2)))
    
        #Now combine 
        train = xr.concat([train_subset1, train_subset2],dim='S').sortby('S')
        train_set = train.sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        
        val_set = file.sel(S=(file['S.year'] >= val_start))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)

        test_set = file.sel(S=(file['S.year'] >= test_start))
        test_set = test_set.sel(S=(test_set['S.year'] <= test_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)

        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_start,train_end,train_start2, train_end2, val_start,val_end,test_start,test_end,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_masked_{test_end}.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_masked_{test_end}.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_masked_{test_end}.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)



def create_stacked_files_by_lead_for_verification_by_individual_grid_cell(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'Data/model_npy_inputs/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        train_set = file.sel(S=(file['S.year'] <= train_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        val_set = file.sel(S=(file['S.year'] > train_end))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        test_set = file.sel(S=(file['S.year'] >= test_start)).sel(L=day_lead).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze().astype(np.float32)
        
        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_by_grid_cell_masked.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_by_grid_cell_masked.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_by_grid_cell_masked.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)


def create_stacked_files_by_lead_for_verification_ensemble_mean(file: xr.DataArray, train_end: int, val_end: int, test_start: int, variable: str, obs_or_forecast: str, region_name: str, init_date_list: list, lead_select: list):
    '''We are stacking the files by lead time. We are going to mask RZSM outside of CONUS with a value of 0 because 
    otherwise the values will be np.nan which is not good for DL algorithms.
    
    For each lead and convert from: (S,M,L,Y,X) to be  (S*M,Y,X).
    Only for observation files.These have data in lead 0 section because the rolling mean was applied to the 3-d timeseries starting from 1999.
    
    This is the VERIFICATION dataset. Only need to do RZSM.
    
    '''
    
    print(f'\nWorking on file {variable} to save as verification observation data. This takes a while because data is not loaded into memory')

    #Save directory for plots
    output_npy_dir = f'Data/model_npy_inputs/{region_name}/Verification_data'
    os.system(f'mkdir -p {output_npy_dir}')
    
    def retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead):
        #restrict to only dates for verification
        file = file.sel(S=slice(init_date_list[0],init_date_list[-1]))
        
        #Split into train/val/test
        train_set = file.sel(S=(file['S.year'] <= train_end)).sel(L=day_lead).mean(dim='M').to_array().squeeze().astype(np.float32)
        val_set = file.sel(S=(file['S.year'] > train_end))
        val_set = val_set.sel(S=(val_set['S.year'] <= val_end)).sel(L=day_lead).mean(dim='M').to_array().squeeze().astype(np.float32)
        test_set = file.sel(S=(file['S.year'] >= test_start)).sel(L=day_lead).mean(dim='M').to_array().squeeze().astype(np.float32)
        
        return(train_set,val_set,test_set)

    
    #All RZSM values already have a value of 0 for where there is not land
    
    #Tmax is not masked anywhere, it has all the values it needs to
    for day_lead in lead_select:
        #Data is subset to 2000-2019, like GEFSv12 inits
        train_set,val_set,test_set = retrieve_stacked_train_val_test(file,train_end,val_end,test_start,day_lead)
        
        print(f'Train observation shape: {train_set.shape}')
        print(f'Validation observation shape: {val_set.shape}')
        print(f'Testing observation shape: {test_set.shape}')
        
        #rename day-lead for 0 day 
        if day_lead==0:
            day_lead=-1 #just to make sure that we name the file properly

        #create a numpy array
        train_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_train_mean_masked.npy'
        val_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_val_mean_masked.npy'
        test_save = f'{output_npy_dir}/{obs_or_forecast}_{variable}_lead_{(day_lead+1)//7}_test_mean_masked.npy'

        '''all np.nan values should already have a zero for RZSM, but just in case fillna here. 
        If np.nan are in any spaces of the dataset, the deep learning algorithm will break'''
        
        np.save(train_save,train_set.fillna(0).values)
        np.save(val_save,val_set.fillna(0).values)
        np.save(test_save,test_set.fillna(0).values)

    return(0)

def final_npy_model_input_directory(region_name: str) -> str:
    npy_dir_for_model_inputs = f'Data/model_npy_input_data/{region_name}'
    os.system(f'mkdir -p {npy_dir_for_model_inputs}')
    return(npy_dir_for_model_inputs)

def final_npy_model_input_directory_TIMESERIES_CLASS(region_name: str) -> str:
    npy_dir_for_model_inputs = f'timeseries/Data/model_npy_input_data/{region_name}/Model_input_data'
    os.system(f'mkdir -p {npy_dir_for_model_inputs}')
    return(npy_dir_for_model_inputs)


def retrieve_stacked_files_and_save_for_model_inputs(file: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, init_date_list: str) -> None:
    print('This is only taking the model inputs which are from observations at daily lags (which are actually in weekly form except the very first day (-1))')

    npy_dir_for_model_inputs = f'{final_npy_model_input_directory(region_name)}/Model_input_data'

    os.makedirs(npy_dir_for_model_inputs,exist_ok=True)

    values = daily_lags = make_daily_lags()
    keys = [f'Lag{i}' for i in values]

    train_dates = [i for i in init_date_list if i < str(train_end+1)]
    test_dates = [i for i in init_date_list if i >= str(test_start)]
    val_dates = [i for i in init_date_list if i <= str(val_end+1)]
    val_dates = [i for i in val_dates if i >= str(val_end-1)]
    
    #Testing if we can load the file into memory before saving because it takes too long to open in later scripts
    print('Loading into memory before we save. This takes a while.')
    #Create a dictionary with the above keys and values for train, val, and test datasets
    len_lags = len(values)
    file = file.isel(L=slice(0,len_lags)).astype(np.float32).load() 


    training_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training.pickle'
    val_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation.pickle'
    test_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing.pickle'

    # Check if files exist before removing them
    if os.path.exists(training_save):
        os.remove(training_save)
    if os.path.exists(val_save):
        os.remove(val_save)
    if os.path.exists(test_save):
        os.remove(test_save)
        
    
    print(f'\nWorking on training data for {variable}')
    # %memit
    out_dict_training = { k:file.sel(S=train_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(training_save,'wb') as handle:
        pickle.dump(out_dict_training,handle)
    
    del out_dict_training
    
    print(f'\nWorking on validation data for {variable}')
    # %memit
    out_dict_validation = { k:file.sel(S=val_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(val_save,'wb') as handle:
        pickle.dump(out_dict_validation,handle)
        
    del out_dict_validation
    
    print(f'\nWorking on testing data for {variable}')
    # %memit
    out_dict_test = { k:file.sel(S=test_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(test_save,'wb') as handle:
        pickle.dump(out_dict_test,handle)
    
    del out_dict_test
    
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))

def retrieve_stacked_files_and_save_for_model_inputs_TIMESERIES_CLASS(file: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, init_date_list: str) -> None:
    print('This is only taking the model inputs which are from observations at daily lags (which are actually in weekly form except the very first day (-1))')

    npy_dir_for_model_inputs = final_npy_model_input_directory_TIMESERIES_CLASS(region_name)

    values = daily_lags = make_daily_lags()
    keys = [f'Lag{i}' for i in values]

    train_dates = [i for i in init_date_list if i < str(train_end+1)]
    test_dates = [i for i in init_date_list if i >= str(test_start)]
    val_dates = [i for i in init_date_list if i <= str(val_end+1)]
    val_dates = [i for i in val_dates if i >= str(val_end-1)]
    
    #Testing if we can load the file into memory before saving because it takes too long to open in later scripts
    print('Loading into memory before we save. This takes a while.')
    #Create a dictionary with the above keys and values for train, val, and test datasets
    len_lags = len(values)
    file = file.isel(L=slice(0,len_lags)).astype(np.float32).load() 
    
    print(f'\nWorking on training data for {variable}')
    # %memit
    out_dict_training = { k:file.sel(S=train_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)
    
    del out_dict_training
    
    print(f'\nWorking on validation data for {variable}')
    # %memit
    out_dict_validation = { k:file.sel(S=val_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)
        
    del out_dict_validation
    
    print(f'\nWorking on testing data for {variable}')
    # %memit
    out_dict_test = { k:file.sel(S=test_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)
    
    del out_dict_test
    
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))



def retrieve_stacked_files_and_save_for_model_inputs_with_different_testing_years(file: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, init_date_list: str, test_end: int, train_start: int, train_start2: int, train_end2: int, val_start:int) -> None:
    print('This is only taking the model inputs which are from observations at daily lags (which are actually in weekly form except the very first day (-1))')

    npy_dir_for_model_inputs = final_npy_model_input_directory(region_name)

    values = daily_lags = make_daily_lags()
    keys = [f'Lag{i}' for i in values]

    #First longest set of dates (e.g., 2000-2010)
    train_dates1 = [i for i in init_date_list if i < str(train_end+1)]
    train_dates2 = [i for i in init_date_list if i >= str(train_start2) and i < str(train_end2+1)]
    train_dates = train_dates1+train_dates2
    
    test_dates = [i for i in init_date_list if i >= str(test_start) and i < str(test_end+1)]
    
    val_dates = [i for i in init_date_list if i >= str(val_start) and i < str(val_end+1)]
    
    #Testing if we can load the file into memory before saving because it takes too long to open in later scripts
    print('Loading into memory before we save. This takes a while.')
    #Create a dictionary with the above keys and values for train, val, and test datasets
    len_lags = len(values)
    file = file.isel(L=slice(0,len_lags)).astype(np.float32).load() 
    
    print(f'\nWorking on training data for {variable}')
    # %memit
    out_dict_training = { k:file.sel(S=train_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)
    
    del out_dict_training
    
    print(f'\nWorking on validation data for {variable}')
    # %memit
    out_dict_validation = { k:file.sel(S=val_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)
        
    del out_dict_validation
    
    print(f'\nWorking on testing data for {variable}')
    # %memit
    out_dict_test = { k:file.sel(S=test_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)
    
    del out_dict_test
    
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))



def retrieve_stacked_files_and_save_for_model_inputs_ensemble_mean(file: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, init_date_list: str) -> None:
    print('This is only taking the model inputs which are from observations at daily lags (which are actually in weekly form except the very first day (-1))')

    npy_dir_for_model_inputs = final_npy_model_input_directory(region_name)

    values = daily_lags = make_daily_lags()
    keys = [f'Lag{i}' for i in values]

    train_dates = [i for i in init_date_list if i < str(train_end+1)]
    test_dates = [i for i in init_date_list if i >= str(test_start)]
    val_dates = [i for i in init_date_list if i <= str(val_end+1)]
    val_dates = [i for i in val_dates if i >= str(val_end-1)]
    
    #Testing if we can load the file into memory before saving because it takes too long to open in later scripts
    print('Loading into memory before we save. This takes a while.')
    #Create a dictionary with the above keys and values for train, val, and test datasets
    len_lags = len(values)
    file = file.isel(L=slice(0,len_lags)).astype(np.float32).load() 
    
    print(f'\nWorking on training data for {variable}')
    # %memit
    out_dict_training = { k:file.sel(S=train_dates).sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_training.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)
    
    del out_dict_training
    
    print(f'\nWorking on validation data for {variable}')
    # %memit
    out_dict_validation = { k:file.sel(S=val_dates).sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_validation.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)
        
    del out_dict_validation
    
    print(f'\nWorking on testing data for {variable}')
    # %memit
    out_dict_test = { k:file.sel(S=test_dates).sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_testing.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)
    
    del out_dict_test
    
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))


def retrieve_stacked_files_and_save_for_model_inputs_by_individual_grid_cell(file: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, init_date_list: str) -> None:
    print('This is only taking the model inputs which are from observations at daily lags (which are actually in weekly form except the very first day (-1))')

    npy_dir_for_model_inputs = final_npy_model_input_directory(region_name)

    values = daily_lags = make_daily_lags()
    keys = [f'Lag{i}' for i in values]

    train_dates = [i for i in init_date_list if i < str(train_end+1)]
    test_dates = [i for i in init_date_list if i >= str(test_start)]
    val_dates = [i for i in init_date_list if i <= str(val_end+1)]
    val_dates = [i for i in val_dates if i >= str(val_end-1)]
    
    #Testing if we can load the file into memory before saving because it takes too long to open in later scripts
    print('Loading into memory before we save. This takes a while.')
    #Create a dictionary with the above keys and values for train, val, and test datasets
    len_lags = len(values)
    file = file.isel(L=slice(0,len_lags)).astype(np.float32).load() 
    
    print(f'\nWorking on training data for {variable}')
    # %memit
    out_dict_training = { k:file.sel(S=train_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_by_grid_cell_training.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)
    
    del out_dict_training
    
    print(f'\nWorking on validation data for {variable}')
    # %memit
    out_dict_validation = { k:file.sel(S=val_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)
        
    del out_dict_validation
    
    print(f'\nWorking on testing data for {variable}')
    # %memit
    out_dict_test = { k:file.sel(S=test_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)
    
    del out_dict_test
    
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))

def stack_reforecasts(var_min_max: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int,lead_select: list) -> None:
    print(f'\nStacking together REFORECAST variable {variable}')
    print('Loading file into memory. This takes a while.')

    npy_dir_for_model_inputs = f"{final_npy_model_input_directory(region_name)}/Model_input_data"
    
    var_min_max = var_min_max.astype(np.float32).load()
    
    train_set = var_min_max.sel(S=(var_min_max['S.year'] <= train_end))
    val_set = var_min_max.sel(S=(var_min_max['S.year'] > train_end))
    val_set = val_set.sel(S=(val_set['S.year'] <= val_end))
    test_set = var_min_max.sel(S=(var_min_max['S.year'] >= test_start))
    # test_set.RZSM[0,0,7,:,:].values

    #select leads 0-5 and saves as a dictionary
    
    keys = ['Lead1','Lead2','Lead3','Lead4','Lead5']
    values = lead_select
    
    #Create a dictionary with the above keys and values for train, val, and test datasets
    out_dict_training = { k:train_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_validation = { k:val_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_test = { k:test_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}

    
    training_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training.pickle'
    val_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation.pickle'
    test_save = f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing.pickle'

    if os.path.exists(training_save):
        os.remove(training_save)
    if os.path.exists(val_save):
        os.remove(val_save)
    if os.path.exists(test_save):
        os.remove(test_save)
        
    
    with open(training_save,'wb') as handle:
        pickle.dump(out_dict_training,handle)

    del out_dict_training
    
    with open(val_save,'wb') as handle:
        pickle.dump(out_dict_validation,handle)

    del out_dict_validation
    
    with open(test_save,'wb') as handle:
        pickle.dump(out_dict_test,handle)

    del out_dict_test
        
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))

def stack_reforecasts_TIMESERIES_CLASS(var_min_max: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int,lead_select: list) -> None:
    print(f'\nStacking together REFORECAST variable {variable}')
    print('Loading file into memory. This takes a while.')

    npy_dir_for_model_inputs = final_npy_model_input_directory_TIMESERIES_CLASS(region_name)
    
    var_min_max = var_min_max.astype(np.float32).load()
    
    train_set = var_min_max.sel(S=(var_min_max['S.year'] <= train_end))
    val_set = var_min_max.sel(S=(var_min_max['S.year'] > train_end))
    val_set = val_set.sel(S=(val_set['S.year'] <= val_end))
    test_set = var_min_max.sel(S=(var_min_max['S.year'] >= test_start))
    # test_set.RZSM[0,0,7,:,:].values

    #select leads 0-5 and saves as a dictionary
    
    keys = ['Lead1','Lead2','Lead3','Lead4','Lead5']
    values = lead_select
    
    #Create a dictionary with the above keys and values for train, val, and test datasets
    out_dict_training = { k:train_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_validation = { k:val_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_test = { k:test_set.sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
       
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)

    del out_dict_training
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)

    del out_dict_validation
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)

    del out_dict_test
        
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))


def stack_reforecasts_with_different_testing_years(var_min_max: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int, lead_select: list, train_start: int, train_start2: int, train_end2: int, val_start:int, test_end: int, init_date_list: list) -> None:
    print(f'\nStacking together REFORECAST variable {variable}')
    print('Loading file into memory. This takes a while.')

    npy_dir_for_model_inputs = final_npy_model_input_directory(region_name)
    
    var_min_max = var_min_max.astype(np.float32).load()

    #First longest set of dates (e.g., 2000-2010)
    train_dates1 = [i for i in init_date_list if i < str(train_end+1)]
    train_dates2 = [i for i in init_date_list if i >= str(train_start2) and i < str(train_end2+1)]
    train_dates = train_dates1+train_dates2
    
    test_dates = [i for i in init_date_list if i >= str(test_start) and i < str(test_end+1)]
    
    val_dates = [i for i in init_date_list if i >= str(val_start) and i < str(val_end+1)]
    

    # test_set.RZSM[0,0,7,:,:].values

    #select leads 0-5 and saves as a dictionary
    
    keys = ['Lead1','Lead2','Lead3','Lead4','Lead5']
    values = lead_select
    
    #Create a dictionary with the above keys and values for train, val, and test datasets
    out_dict_training = { k:var_min_max.sel(S=train_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_validation = { k:var_min_max.sel(S=val_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_test = { k:var_min_max.sel(S=test_dates).sel(L=v).stack(combine_models=['S','M']).transpose('combine_models','Y','X').to_array().squeeze() for (k,v) in zip(keys, values)}
       
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_training_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)

    del out_dict_training
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_validation_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)

    del out_dict_validation
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_testing_{test_end}.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)

    del out_dict_test
        
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))


def stack_reforecasts_ensemble_mean(var_min_max: xr.DataArray, variable: str, obs_or_forecast: str, region_name: str, train_end: int, val_end: int, test_start: int,lead_select: list) -> None:
    print(f'\nStacking together REFORECAST variable {variable}')
    print('Loading file into memory. This takes a while.')

    npy_dir_for_model_inputs = final_npy_model_input_directory(region_name)
    
    var_min_max = var_min_max.astype(np.float32).load()
    
    train_set = var_min_max.sel(S=(var_min_max['S.year'] <= train_end))
    val_set = var_min_max.sel(S=(var_min_max['S.year'] > train_end))
    val_set = val_set.sel(S=(val_set['S.year'] <= val_end))
    test_set = var_min_max.sel(S=(var_min_max['S.year'] >= test_start))
    # test_set.RZSM[0,0,7,:,:].values

    #select leads 0-5 and saves as a dictionary
    
    keys = ['Lead1','Lead2','Lead3','Lead4','Lead5']
    values = lead_select
    
    #Create a dictionary with the above keys and values for train, val, and test datasets
    out_dict_training = { k:train_set.sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_validation = { k:val_set.sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
    out_dict_test = { k:test_set.sel(L=v).mean(dim='M').to_array().squeeze() for (k,v) in zip(keys, values)}
       
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_training.pickle','wb') as handle:
        pickle.dump(out_dict_training,handle)

    del out_dict_training
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_validation.pickle','wb') as handle:
        pickle.dump(out_dict_validation,handle)

    del out_dict_validation
    
    with open(f'{npy_dir_for_model_inputs}/{obs_or_forecast}_{variable}_mean_testing.pickle','wb') as handle:
        pickle.dump(out_dict_test,handle)

    del out_dict_test
        
    return(print(f'Saved {variable} data into {npy_dir_for_model_inputs}'))



def convert_OBS_anomaly_to_SubX_format(init_date_list: str, region_name: str, anom_mean_subset: xr.DataArray, fcst_dir: str, lead_select: list, rzsm_source: str) -> None:  
            
    '''We are going to create new leads that are different than reforecast. The reasoning for this is that we want the actual weekly lags (and 1 day lag) and this will
    assist with future predictions within the deep learning model'''

    anomaly_file = anom_mean_subset.sel(L=lead_select)
    ref_dir = f'{fcst_dir}/soilw_bgrnd'
        
    save_dir = f'Data/reanalysis/{rzsm_source}/{region_name}/RZSM_anomaly_reformat_SubX_format'
    os.system(f'mkdir -p {save_dir}')
    
    template = sorted(glob(f'{ref_dir}/*.n*'))[0]
    template = xr.open_dataset(template).sel(L=lead_select)

    print(f'\nAdding data within function convert_OBS_anomaly_to_SubX_format to file for leads {lead_select}')
    
    for _date in init_date_list:
        
        _date = pd.to_datetime(_date)
        save_date = f'{_date.year}-{_date.month:02}-{_date.day:02}'
        obs_file_name = f'RZSM_anomaly_reformat_{save_date}.nc4'
        save_file = f'{save_dir}/{obs_file_name}'
        
        if not os.path.exists(save_file):
            #Grab a single SubX to use as the template. Doesn't matter if it is the same variable or not or the same date
            open_date_SubX = template
            out_file = xr.zeros_like(open_date_SubX)
            
            '''We are going to create a new lead day that represents the previous day before the forecast was initialized
            #New shape will be (1x11x48xlatxlon)
            This will include the day lag 1, and weekly lags 1-12'''
            
            file_shape = out_file[xarray_varname(out_file)].shape

            # os.system(f'rm {save_file}') #Just to avoid getting random duplicates
            print(f'Working on initialized day {_date} to find values integrating with SubX models, leads, & coordinates and saving data into {save_dir}.')

            out_file[xarray_varname(out_file)][0,:, :, :, :] = \
                anomaly_file[xarray_varname(anomaly_file)].sel(S = _date).values

            var_OUT = xr.Dataset(
                data_vars = dict(
                    var = (['S','M','L','Y','X'], out_file[xarray_varname(out_file)].values),
                ),
                coords = dict(
                    S = np.atleast_1d(_date),
                    X = open_date_SubX.X.values,
                    Y = open_date_SubX.Y.values,
                    L = out_file.L.values,
                    M = open_date_SubX.M.values,
    
                ),
                attrs = dict(
                    Description = f'RZSM anomaly values on the exact same date and grid \
                    cell as EMC reforecast data. 7-day rolling mean already applied.'),
            )                    
    
            var_OUT = var_OUT.astype(np.float32)
            var_OUT.to_netcdf(save_file)

    return(0)


def return_rmse_and_mae_pickle_files(file_rmse_save, file_mae_save, completed_mae, completed_rmse):
    try:
        with open(file_rmse_save, "rb") as f:
            rmse_output = pickle.load(f)
    
        with open(file_mae_save, "rb") as f:
            mae_output = pickle.load(f)    

        with open(completed_mae, 'rb') as f:
            mae_complete = pickle.load(f) 
            
        with open(completed_rmse, 'rb') as f:
            rmse_complete = pickle.load(f) 
            
    except FileNotFoundError:
        rmse_output = {}
        rmse_output['OBS'] = {}
        rmse_output['HYB'] = {}
    
        mae_output = {}
        mae_output['OBS'] = {}
        mae_output['HYB'] = {}

        rmse_complete = {}
        rmse_complete['OBS'] = {}
        rmse_complete['HYB'] = {}
    
        mae_complete= {}
        mae_complete['OBS'] = {}
        mae_complete['HYB'] = {}
        
    return(rmse_output, mae_output, rmse_complete, mae_complete)


def check_if_already_completed_permuatation(rmse_complete, mae_complete, ex_num,  OBS, HYB, channel, model_name):
    #First check if the model name exists
    try:
        #If exists, then see if the channel is already in file
        if ex_num in OBS:
            rmse_check = 'Pass' if channel in rmse_complete['OBS'][model_name] else 'Fail'
            mae_check = 'Pass' if channel in mae_complete['OBS'][model_name] else 'Fail'
        else:
            rmse_check = 'Pass' if channel in rmse_complete['HYB'][model_name] else 'Fail'
            mae_check = 'Pass' if channel in mae_complete['HYB'][model_name] else 'Fail'
            
        overall_check = 'Pass' if (rmse_check == 'Pass') and (mae_check == 'Pass') else 'Fail'
    except KeyError:
        print('This has not been completed yet')
        overall_check = 'Fail'

    if overall_check == 'Pass':
        return('Completed')
    else:
        return('Not-Completed')


def append_to_complete_list(complete_file, model_name, channel, hyb_or_obs):
    try:
        complete_file[hyb_or_obs][model_name].append(channel)
    except KeyError:
        complete_file[hyb_or_obs][model_name] = [channel]
    return(complete_file)


def expand_dims_by_lead(file):
    return(file.expand_dims({'L':1}).transpose(*dim_order))

def select_data_by_lead(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only, day_num):
    obs_lead = expand_dims_by_lead(obs_anomaly_SubX_format.sel(L=day_num))
    baseline_gefs =  expand_dims_by_lead(baseline_anomaly.sel(L=day_num))
    baseline_ecm =  expand_dims_by_lead(baseline_ecmwf.sel(L=day_num))
    var_OUT = var_OUT
    template =  expand_dims_by_lead(template_testing_only.sel(L=day_num))
    return(obs_lead,baseline_gefs,baseline_ecm,var_OUT,template)