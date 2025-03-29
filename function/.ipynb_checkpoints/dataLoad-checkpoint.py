#!/user/bin/env python3
from . import conf
import xarray as xr
from . import preprocessUtils as putils
from . import verifications
import pandas as pd
from glob import glob
import numpy as np
import pickle
from function import masks

def load_rzsm_observations(soil_dir, region_name):
    obs_original = xr.open_dataset(f'{soil_dir}/{region_name}/RZSM_anomaly.nc')
    name = putils.xarray_varname(obs_original)
    obs_original = obs_original.rename({name:'RZSM'}).drop('season').load()
    obs_raw = xr.open_dataset(f'{soil_dir}/{region_name}/RZSM_weighted_mean_0_100cm.nc4')
    obs_raw = obs_raw.rename({name:'RZSM'}).load()
    return obs_original, obs_raw
    

def return_init_and_testing_dates(region_name,test_start,test_end):
    init_dates = putils.get_init_date_list(f'{conf.gefsv12_data}/{region_name}/soilw_bgrnd')
    dt_dates = [pd.to_datetime(i) for i in init_dates]
    if test_end == None:
        only_testing_dates = [i for i in dt_dates if i >= pd.to_datetime(test_start)]
    else:
        only_testing_dates = [i for i in dt_dates if i >= pd.to_datetime(test_start) and i <= pd.to_datetime(test_end)]
    
    return init_dates,dt_dates,only_testing_dates



def load_GEFS_soil_reforecast(region_name,mask_anom):
    raw_var = putils.return_reforecast_files_by_concatenation(dir_path = f"{conf.gefsv12_data}/{region_name}", name_of_var = 'soilw_bgrnd', region_name = region_name)
    baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/soil*nc'))
    baseline_anomaly_climatology = xr.open_mfdataset(baseline_anomaly_file_list).astype(np.float32).load()
    baseline_anomaly_climatology = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly_climatology)
    return raw_var, baseline_anomaly_file_list, baseline_anomaly_climatology
    

def load_ECMWF_soil_reforecast(region_name,mask_anom, init_dates):
    ecmwf_files_all = sorted(glob(f'{conf.ecmwf_data}/{region_name}/soilw_bgrnd/soil*'))
    correct_ecmwf_files = [i for i in ecmwf_files_all if any(init_date in i for init_date in init_dates)]
    raw_var = xr.open_mfdataset(correct_ecmwf_files)
   
    baseline_anomaly_climatology = verifications.load_ECMWF_baseline_anomaly(region_name)
    return raw_var, baseline_anomaly_climatology


def load_GEFS_soil_raw(region_name):
    raw_var = putils.return_reforecast_files_by_concatenation(dir_path = f"{conf.gefsv12_data}/{region_name}", name_of_var = 'soilw_bgrnd', region_name = region_name)
    return raw_var
        

def open_ACC_pickle_season(acc_value_directory, region_name, obs_source, week_lead):
    # Open the pickle file in read-binary mode
    with open(f'{acc_value_directory}/Wk_{week_lead}/ACC_vals_{obs_source}_season.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def open_obs_and_baseline_and_bias_corrected_all_leads(region_name, week_lead, start_, end_, mask_anom, test_start, test_end, obs_source,soil_dir):

    print(f'Loading template data from {test_start} to {test_end}')
    mask,mask_anom = masks.load_mask_vals(region_name)

    obs_anomaly_SubX_format_testing =xr.open_mfdataset(f'{soil_dir}/{region_name}/RZSM_anomaly_reformat_SubX_format/RZSM_anomaly*').sel(S=slice(test_start, test_end)).sel(L=week_lead).load()

    template_testing_only = obs_anomaly_SubX_format_testing.copy(deep=True)

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
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=week_lead).sel(S=slice(test_start,test_end)).load()
    
    else:
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=week_lead).sel(S=slice(test_start,test_end)).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)
    
    baseline_ecmwf_file_list = sorted(glob(f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly/*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=week_lead).sel(S=slice(test_start,test_end)).load()

    ecmwf_bias_corrected = xr.open_dataset(conf.return_bias_corrected_anomaly(region_name, 'ECMWF', obs_source)).isel(lead=week_lead).rename({'init':'S','member':'M','lead': 'L','lat':'Y','lon':'X'})
    gefs_bias_corrected = xr.open_dataset(conf.return_bias_corrected_anomaly(region_name, 'GEFSv12', obs_source)).isel(lead=week_lead).rename({'init':'S','member':'M','lead': 'L','lat':'Y','lon':'X'})

    # ecmwf_bias_corrected['L'] = [6,13,20,27,34]
    # gefs_bias_corrected['L'] = [6,13,20,27,34]
    
    # ecmwf_bias_corrected=ecmwf_bias_corrected.sel(L=week_lead).sel(S=slice(test_start,test_end)).load()
    # gefs_bias_corrected=gefs_bias_corrected.sel(L=week_lead).sel(S=slice(test_start,test_end)).load()
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format_testing, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only,ecmwf_bias_corrected,gefs_bias_corrected)



def return_obs_percentiles(region_name, test_start, test_end, testing_dist,soil_dir):    
    if testing_dist:
        save_percentile_observations = f'{conf.gleam_data}/{region_name}/anomaly_percentile_RZSM_full_distribution_with_different_thresholds_testing_distribution.nc4'
        obs_anomaly_split_percentiles = xr.open_dataset(save_percentile_observations)
        obs_anom_percentile_SubX_format = xr.open_mfdataset(f'{soil_dir}/{region_name}/RZSM_percentile_reformat_testing_distribution/*').sel(S=slice(test_start,test_end)).load()
    else:
        save_percentile_observations = f'{conf.gleam_data}/{region_name}/anomaly_percentile_RZSM_full_distribution_with_different_thresholds.nc4'
        obs_anomaly_split_percentiles = xr.open_dataset(save_percentile_observations)
        obs_anom_percentile_SubX_format = xr.open_mfdataset(f'{soil_dir}/{region_name}/RZSM_percentile_reformat/*').sel(S=slice(test_start,test_end)).load()
    
    return obs_anomaly_split_percentiles, obs_anom_percentile_SubX_format


def load_ECMWF_percentile_anomaly(region_name: str, testing_dist, test_start, test_end) -> xr.DataArray:
    
    if testing_dist:
        print("Loading ECMWF MEM soil moisture percentile files from testing distribution (these are bias corrected)")
        files = sorted(glob(f'{conf.ecmwf_data}/{region_name}/soilw_bgrnd/percentiles_MEM_testing_distribution/*'))
    else:
        print("Loading ECMWF MEM soil moisture percentile files. Non-bias corrected data")
        files = sorted(glob(f'{conf.ecmwf_data}/{region_name}/soilw_bgrnd/percentiles_MEM/*'))
    datasets = [xr.open_dataset(f) for f in files]
    return xr.concat(datasets, dim="S").sel(S=slice(test_start,test_end))

# def load_GEFSv12_percentile_anomaly(region_name: str) -> xr.DataArray:
#     if region_name == 'CONUS':
#         return(xr.open_mfdataset(f'Data/GEFSv12_reforecast/soilw_bgrnd/percentiles_MEM/*', combine='nested', concat_dim='S'))
#     else:
#         return(xr.open_mfdataset(f'Data_{region_name}/GEFSv12_reforecast/soilw_bgrnd/percentiles_MEM/*', combine='nested', concat_dim='S'))

def load_GEFSv12_percentile_anomaly(region_name: str, testing_dist, test_start, test_end) -> xr.DataArray:

    if testing_dist:
        print("Loading GEFS MEM soil moisture percentile files from testing distribution (these are bias corrected)")
        files = sorted(glob(f'{conf.gefsv12_data}/{region_name}/soilw_bgrnd/percentiles_MEM_testing_distribution/*'))
    else:
        print("Loading GEFSv12 MEM soil moisture percentile files. Non-bias corrected data")
        files = sorted(glob(f'{conf.ecmwf_data}/{region_name}/soilw_bgrnd/percentiles_MEM/*'))
    datasets = [xr.open_dataset(f) for f in files]
    return xr.concat(datasets, dim="S").sel(S=slice(test_start,test_end))