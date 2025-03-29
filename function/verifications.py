#!/usr/bin/env python3
import xarray as xr
import numpy as np
import climpred
from xclim import sdba
from glob import glob
import os
import random
import pandas as pd

#Personal functions
from . import preprocessUtils as putils
from . import funs as f
from . import conf


def return_experiment_colors_and_names():
    BC = ['EX0','EX13'] # bias-correction
    OBS = ['EX14','EX15','EX16','EX17','EX22','EX23','EX24','EX25'] # obs.-driven
    HYB = ['EX1','EX2','EX3','EX4','EX5','EX6','EX7','EX8','EX9','EX10','EX11','EX12',
           'EX18','EX19','EX20','EX21','EX27','EX28','EX29'] # hybrid
    
    print('Bias-correction (DM-BC_DL) is black')
    print(f'DM-BC_DL experiments is {print(BC)}\n')
    print('Observation-driven (DL) is red')
    print(f'DL experiments is {print(OBS)}\n')
    print('Hybrid (DL-DM)')
    print(f'DL-DM experiments is {print(HYB)}\n')
    
    return(BC,OBS,HYB)


def load_UNET_files_with_mean(gefs, file, region_name, day_num,new_source,test_year):
    add_to_file = gefs.copy(deep = True)
    test_name = file.split('testing_')[-1].split('.npy')[0].split('ensemble_')[-1]
    load_ = np.expand_dims(np.load(file),-1)
    load_.shape
    load_ = np.reshape(load_,(load_.shape[0],load_.shape[-2], load_.shape[-1], load_.shape[1], load_.shape[2]))
    load_.shape
    load_ = np.where(load_ == 0,np.nan,load_)

    load_ = reverse_min_max_scaling(load_, region_name, day_num,new_source,test_year)
    add_to_file[putils.xarray_varname(add_to_file)][:,:,:,:,:] = load_

    return(add_to_file)

def load_UNET_files(gefs, file, region_name, day_num,new_source,test_year):
    '''This only grabs a single week and just loads into a much larger dataset'''
    '''This could be re-worked later'''
    print("Loading UNET testing predictions and reversing the min-max scaling to be back to anomalies")
    add_to_file = gefs.copy(deep=True)
    test_name = file.split('testing_')[-1].split('.npy')[0]
    
    load_ = np.load(file)[-1,:,:,:,0] 
    test = reverse_min_max_scaling(load_, region_name, day_num,new_source,test_year)
    
    test = np.reshape(test,(test.shape[0]//11,11,test.shape[1],test.shape[2]))
    test = np.expand_dims(test, -1)
    #Now re-order the dimensions to match SubX
    load_ =  np.reshape(test,(test.shape[0], test.shape[1], test.shape[-1], test.shape[2], test.shape[3]))
    
    add_to_file[putils.xarray_varname(add_to_file)][:,:,:,:,:] = load_

    return(add_to_file)

def load_UNET_all_weeks(gefs, files, region_name, new_source,test_year,experiment_number,mask_anom):
    print("Loading UNET testing predictions and reversing the min-max scaling to be back to anomalies")
    add_to_file = gefs.copy(deep=True)

    for idx,file in enumerate(files):
        day_num = ((idx+1) * 7) - 1  # Lead time in days
        print(f'Day number is {day_num}')
    
        load_ = np.load(file)[-1,:,:,:,0] 
        test = reverse_min_max_scaling(load_, region_name, day_num,new_source,test_year)
        test = np.reshape(test,(test.shape[0]//11,11,test.shape[1],test.shape[2]))
        test = np.expand_dims(test, -1)
        
        load_ =  np.reshape(test,(test.shape[0], test.shape[1], test.shape[-1], test.shape[2], test.shape[3]))
        print(load_.shape)
        load_ = np.where(np.isnan(mask_anom),np.nan,load_)
        
        add_to_file[putils.xarray_varname(add_to_file)][:,:,idx,:,:] = load_[:,:,0,:,:]

    return(add_to_file)


def load_XGBoost_file_and_make_ensemble_spread(gefs, file, source_of_XGBoost_reforecast,day_num, region_name, test_year):

    '''We only trained the ensemble mean as the prediction. So we only have 1 value, now we can make 11 values just for the climpred function ACC'''
    add_to_file = gefs.copy(deep = True)
    # break
    #Still working here
    test_name = file.split('testing_')[-1].split('.npy')[0]
    load_ = np.expand_dims(np.load(file),-1)
    load_.shape
    load_ = np.where(load_ == 0,np.nan,load_)
    add_realizations = np.empty(shape=(load_.shape[0],11,load_.shape[1],load_.shape[2],load_.shape[3])) #This will help with climpred functions
    for j in range(11):
        add_realizations[:,j,:,:,:] = load_
    add_realizations = reverse_min_max_scaling(add_realizations, region_name, day_num, source_of_XGBoost_reforecast,test_year)
    add_realizations.shape
    add_realizations =  np.reshape(add_realizations,(add_realizations.shape[0], add_realizations.shape[1], add_realizations.shape[-1], add_realizations.shape[2], add_realizations.shape[3])) #Reshape to match gefs/ecmwf
    add_to_file[putils.xarray_varname(add_to_file)][:,:,:,:,:] = add_realizations

    return(add_to_file)

# def load_RZSM_anomaly_obs(region_name: str) -> xr.DataArray:
#     return(xr.open_mfdataset(f'{conf.gleam_data}/{region_name}/RZSM_anomaly_reformat_SubX_format/*'))

def load_RZSM_anomaly_obs(region_name: str, soil_dir) -> xr.DataArray:
    print(f"Loading {soil_dir} baseline anomaly files")
    files = sorted(glob(f'{soil_dir}/{region_name}/RZSM_anomaly_reformat_SubX_format/*.nc4'))
    datasets = [xr.open_dataset(f) for f in files]
    return xr.concat(datasets, dim="S")

# def load_ECMWF_baseline_anomaly(region_name: str) -> xr.DataArray:
#     return(xr.open_mfdataset(f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly/*'))

def load_ECMWF_baseline_anomaly(region_name: str) -> xr.DataArray:
    print("Loading ECMWF baseline anomaly files")
    files = sorted(glob(f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly/*.nc'))
    datasets = [xr.open_dataset(f) for f in files]
    return xr.concat(datasets, dim="S").load()

# def load_GEFSv12_baseline_anomaly(region_name: str) -> xr.DataArray:
#     return(xr.open_mfdataset(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/*'))

def load_GEFSv12_baseline_anomaly(region_name: str) -> xr.DataArray:
    print("Loading GEFSv12 baseline anomaly files")
    files = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/*.nc'))
    datasets = [xr.open_dataset(f) for f in files]
    return xr.concat(datasets, dim="S").load()



# def load_ECMWF_percentile_anomaly(region_name: str) -> xr.DataArray:
#     return(xr.open_mfdataset(f'{conf.ecmwf_data}/{region_name}/soilw_bgrnd/percentiles_MEM/*', combine='nested', concat_dim='S'))




def reverse_min_max_scaling(file,region_name,lead,source,test_year):
    if test_year == 2012:
        add_ = f"_{test_year}"
    elif test_year == 2019:
        add_ = ""

    max_min_source = f'Data/min_max_values/{region_name}'
    
    max_, min_ = np.load(f'{max_min_source}/soilw_bgrnd_{source}_lead{lead}_max{add_}.npy'), np.load(f'{max_min_source}/soilw_bgrnd_{source}_lead{lead}_min{add_}.npy')
    return(file *(max_-min_)+min_)

def reverse_min_max_scaling_for_permutations(file,region_name,lead,source,test_year,var_):
    if test_year == 2012:
        add_ = f"_{test_year}"
    elif test_year == 2019:
        add_ = ""
        
    max_min_source = f'Data/min_max_values/{region_name}'
    
    max_, min_ = np.load(f'{max_min_source}/{var_}_{source}_lead{lead}_max{add_}.npy'), np.load(f'{max_min_source}/{var_}_{source}_lead{lead}_min{add_}.npy')
    return(file *(max_-min_)+min_)


def create_reforecast_with_predictions_single_lead(template_testing_only: xr.DataArray, day_num: int, week_lead: int, experiment_name: str, region_name: str, mask_anom: np.ndarray, start_: str, end_: str, source: str, test_year: int, obs_source) -> xr.DataArray:
    #Load previous predictions from experiments
    temp_cp = template_testing_only.copy(deep=True).sel(L=day_num)

    print(f"Loading prediction predictions/{region_name}/Wk{week_lead}_testing/Wk{week_lead}_testing_{experiment_name}.npy")
    test = reverse_min_max_scaling(np.load(f'predictions/{region_name}/Wk{week_lead}_testing/Wk{week_lead}_testing_{experiment_name}.npy')[2,:,:,:,0],region_name,day_num,obs_source,test_year)

    
    test = np.reshape(test,(test.shape[0]//11,11,test.shape[1],test.shape[2]))

    #Apply mask 
    if region_name == 'CONUS':
        test = np.where(mask_anom == 1, test, np.nan)
    elif region_name in ['australia','china']:
        test = np.where(np.isnan(mask_anom), np.nan, test)
    
    #Add data to file
    temp_cp[putils.xarray_varname(temp_cp)][:,:,:,:] = test

    #Mask the Southeast 
    # temp_cp = temp_cp.sel(X=slice(southeast_lon_left,southeast_lon_right)).sel(Y=slice(southeast_lat_top,southeast_lat_bottom)).mean(dim='M')
    temp_cp = temp_cp.sel(S=slice(start_,end_))
    
    return(temp_cp)

def create_reforecast_with_predictions_RESIDUALS(template_testing_only: xr.DataArray, day_num: int, week_lead: int, experiment_name: str, region_name: str, mask_anom: np.ndarray, start_: str, end_: str, source: str, test_year: int) -> xr.DataArray:
    #Load previous predictions from experiments
    temp_cp = template_testing_only.copy(deep=True).sel(L=day_num)

    test = np.load(f'predictions/{region_name}/Wk{week_lead}_testing/Wk{week_lead}_testing_{experiment_name}.npy')
    test = np.reshape(test,(test.shape[0]//11,11,test.shape[1],test.shape[2]))

    test = reverse_min_max_scaling(test,region_name,day_num,'residuals',test_year)

    test = np.where(mask_anom == 1, test, np.nan)
    
    #Add data to file
    temp_cp.RZSM[:,:,:,:] = test

    #Mask the Southeast 
    # temp_cp = temp_cp.sel(X=slice(southeast_lon_left,southeast_lon_right)).sel(Y=slice(southeast_lat_top,southeast_lat_bottom)).mean(dim='M')
    temp_cp = temp_cp.sel(S=slice(start_,end_))
    
    return(temp_cp)


def anomaly_correlation_coefficient_function_ensemble_mean(var_OUT: np.ndarray, forecast_converted: np.ndarray, obs_converted: np.ndarray) -> np.ndarray:
    '''I put this function into the loop (tried numba, didn't work well) 
    Source ACC:
    https://metclim.ucd.ie/wp-content/uploads/2017/07/DeterministicSkillScore.pdf
    def ACC(FC_anom,OBS_anom):
        top = np.nanmean(FC_anom*OBS_anom) #all forecast anomalies * all observation anomalies                    
        bottom = np.sqrt(np.nanmean(FC_anom**2)*np.nanmean(OBS_anom**2)) #variance of forecast anomalies * variance of observation anomalies
        ACC = top/bottom
        return (ACC)
    '''

    # #Now find pearson correlation by model, lead, and lat/lon
    # for model in range(var_OUT.shape[0]):
    # # for model in range(3,4): for testing
    #     # print(f'Working on model {model+1} for pearson r correlation')

    for Y in range(var_OUT.shape[0]):
        # print(f"Working on latitude index {Y} out of {var_OUT.Y.shape[0]}")
        for X in range(var_OUT.shape[1]):

            '''There is a Zero division error that occurs, to fix this (because numba doesn't like it)
            just check and see if the two files have all 0s or np.nans'''

            # ACC from function
            obs = obs_converted[:, 0, Y, X]
            try:
                subx = forecast_converted[:, 0, Y, X]
            except IndexError:
                subx = forecast_converted[:, Y, X]

            top = np.nanmean(subx*obs)

            bottom = np.sqrt(np.nanmean(subx**2)*np.nanmean(obs**2))
            
            ACC = top/bottom

            var_OUT[Y, X] = ACC

    return(var_OUT)

def rename_subx_for_climpred(file):
    # https://climpred.readthedocs.io/en/stable/examples/subseasonal/daily-subx-example.html
    file = file.rename(S='init')
    file = file.rename(L='lead')
    file["lead"].attrs = {"units": "days"}
    file = file.rename(M='member')
    file = file.rename(X='lon')
    file = file.rename(Y='lat')
    file = file.assign_attrs(lead='days')
    try:
        file = file.drop('time')
        file = file.drop('season')
    except ValueError:
        pass
    return(file)




def rename_obs_for_climpred(file):
    file = file.rename(longitude='lon')
    file = file.rename(latitude='lat')
    # file = file.drop('S')

    # file = set_integer_time_axis(file,time_dim='time')
    # file=file.rename(time='init')
    return(file)

def create_climpred_CRPS(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="crps", comparison="m2o", dim="member",alignment="same_inits").rename({fcst_name:'crps'}).load())

def create_climpred_CRPSS(fcst,obs, comparison):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="crpss", comparison="m2o", dim="member",alignment="same_inits").rename({fcst_name:'crpss'}).load())

# def create_climpred_CRPSS_bootstrap(fcst,obs):
#     fcst_name = list(fcst.keys())[0]
#     object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
#     skill = object_.bootstrap(
#     metric="crpss",
#     comparison="m2o",
#     dim="member",
#     alignment="same_inits",
#     iterations=250
# ).rename({fcst_name: "crpss"}).load()
#     return(skill)

def create_climpred_CRPSS_bootstrap(fcst, obs, num_iterations, metric_):
    fcst_name = list(fcst.keys())[0]

    # Ensure lead units are correctly set
    if 'units' not in fcst['lead'].attrs or fcst['lead'].attrs['units'] != 'days':
        fcst['lead'].attrs['units'] = 'days'

    # Ensure 'init' is a dimension
    if 'init' not in fcst.dims and 'init' in fcst.coords:
        fcst = fcst.swap_dims({'time': 'init'})
        fcst = fcst.set_index(init='init')

    if 'init' not in obs.dims and 'init' in obs.coords:
        obs = obs.swap_dims({'time': 'init'})
        obs = obs.set_index(init='init')

    # Ensure 'member' exists
    if 'member' not in fcst.dims:
        fcst = fcst.expand_dims('member')

    # Print dimensions for debugging
    print("Forecast dataset dims:", fcst.dims)
    print("Observations dataset dims:", obs.dims)

    hindcast = climpred.HindcastEnsemble(fcst).add_observations(obs)

    skill = hindcast.bootstrap(
        metric=metric_,
        comparison="m2o",
        dim="member",
        alignment="same_inits",
        iterations=num_iterations,
        reference=["persistence", "climatology"],
        resample_dim="member",  # Critical fix!
    ).rename({fcst_name: metric_}).load()

    return skill




def create_climpred_ACC(fcst, obs):
    '''For some reason the data from additive bias correction is incorrectly chunked, so this will fix it. But it's pretty slow'''
    
    fcst_name = list(fcst.keys())[0]
    # Ensure forecast dataset is chunked correctly
    fcst = fcst.chunk({'init': -1, 'lon': 'auto', 'lat': 'auto', 'lead': -1})
    hcast = climpred.HindcastEnsemble(fcst)
    # Ensure observation dataset is chunked consistently
    obs = obs.chunk({'time': -1, 'lon': 'auto', 'lat': 'auto'})
    object_ = hcast.add_observations(obs)
    return object_.verify(metric="acc", comparison="e2o", dim="init", alignment="same_inits").rename({fcst_name: 'acc'}).load()


def create_climpred_ACC_no_chunk(fcst, obs):
    '''For some reason the data from additive bias correction is incorrectly chunked, so this will fix it. But it's pretty slow'''
    
    fcst_name = list(fcst.keys())[0]
    # Ensure forecast dataset is chunked correctly
    hcast = climpred.HindcastEnsemble(fcst)
    # Ensure observation dataset is chunked consistently
    object_ = hcast.add_observations(obs)
    return object_.verify(metric="acc", comparison="e2o", dim="init", alignment="same_inits").rename({fcst_name: 'acc'}).load()


def create_climpred_CRPSS_no_chunk(fcst, obs):
    '''For some reason the data from additive bias correction is incorrectly chunked, so this will fix it. But it's pretty slow'''
    fcst_name = list(fcst.keys())[0]
    # Ensure forecast dataset is chunked correctly
    hcast = climpred.HindcastEnsemble(fcst)
    # Ensure observation dataset is chunked consistently
    object_ = hcast.add_observations(obs)
    return object_.verify(metric="crpss", comparison="m2o", dim="member", alignment="same_inits").rename({fcst_name: 'crpss'}).load()

def create_climpred_ACC_persistence(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    persistence = object_.verify(metric="acc", comparison="e2o", dim="init",alignment="same_inits", reference=["persistence","climatology"]).rename({fcst_name:'acc'}).load()
    return(persistence)

def create_climpred_ACC_climatology(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    climatology = object_.verify(metric="acc", comparison="e2o", dim="init",alignment="same_inits", reference=["persistence", "climatology"],skipna=True).rename({fcst_name:'acc'}).load()
    return(climatology)

def create_climpred_crpss_skill_score(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="crpss", comparison="m2o", dim="member",alignment="same_inits").rename({fcst_name:'crpss'}).load())

def create_climpred_crpss_ensemble_spread(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="crpss_es", comparison="m2o", dim="member",alignment="same_inits").rename({fcst_name:'crpss'}).load())


def pos(x):
    return x > 0  # checking binary outcomes

def create_realiability_forecasts(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="reliability", comparison="m2o", dim=["member","init"],alignment="same_verifs",logical=pos).rename({fcst_name:'crpss'}).load())

def create_rank_histogram(fcst,obs):
    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)
    return(object_.verify(metric="rank_histogram", comparison="m2o", dim=["member", "init", "lat", "lon"], alignment="maximize").rename({fcst_name:'rank_histogram'}).load())


def additive_bias_removal(fcst,obs,train_start, train_end):
    metric_kwargs = dict(
    metric="rmse", alignment="maximize", dim="init", comparison="e2o", skipna=True)


    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)

    bias_removed = object_.remove_bias(
            how="additive_mean",
            alignment=metric_kwargs["alignment"],
            train_test_split="fair",
            train_init=slice(train_start, train_end),
        )
    return(bias_removed)


def empirial_quantile_mapping_bias_correction(fcst,obs):
    metric_kwargs = dict(
    metric="rmse", alignment="same_verifs", dim="init", comparison="e2o", skipna=True)


    fcst_name = list(fcst.keys())[0]
    object_ =  climpred.HindcastEnsemble(fcst).add_observations(obs)

    bias_removed = object_.remove_bias(
            how="EmpiricalQuantileMapping",
            alignment=metric_kwargs["alignment"],
            train_test_split="fair",
            train_time=slice("2000", "2015"),
        )
    return(bias_removed)


def open_obs_and_baseline_files_multiple_leads(region_name, leads, test_start, test_end,mask_anom,soil_dir):
    print(f'Loading soil data all the baseline files for observations, GEFSv12, and ECMWF for region {region_name}')

    print(f'Loading {soil_dir} observations')
    obs_anomaly_SubX_format =xr.open_mfdataset(f'{soil_dir}/{region_name}/RZSM_anomaly_reformat_SubX_format/RZSM_anomaly*.nc4').sel(L=leads).astype(np.float32).load()
    obs_anomaly_SubX_format = obs_anomaly_SubX_format.sel(S=slice(test_start,test_end)).load()
    
    template_testing_only = obs_anomaly_SubX_format.copy(deep=True)
    
    var_OUT = np.empty(shape=(obs_anomaly_SubX_format.Y.shape[0], obs_anomaly_SubX_format.X.shape[0])) #48x96
    
    #Mask the final output to be np.nan for ocean values
    var_OUT = np.where(mask_anom==1, np.nan, var_OUT)
    var_OUT[:,:] = 0
    
    #######################################   Reforecast baseline files   ###########################################################################
    # baseline_anomaly_file_list = sorted(glob('Data/GEFSv12_reforecast/soilw_bgrnd/baseline_RZSM_anomaly/RZSM*.nc'))
    print('Loading GEFS data')
    if region_name =='CONUS':
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/soil*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=leads).sel(S=slice(test_start,test_end)).astype(np.float32).load()

    else:
        baseline_anomaly_file_list = sorted(glob(f'{conf.gefsv12_data}/{region_name}/baseline_RZSM_anomaly/soil*.nc'))
        baseline_anomaly = xr.open_mfdataset(baseline_anomaly_file_list).sel(L=leads).sel(S=slice(test_start,test_end)).astype(np.float32).load()
        baseline_anomaly = xr.where(np.isnan(mask_anom),np.nan, baseline_anomaly)

    print('Loading ECMWF data')
    baseline_ecmwf_file_list = sorted(glob(f'{conf.ecmwf_data}/{region_name}/baseline_RZSM_anomaly/soil*.nc'))
    baseline_ecmwf = xr.open_mfdataset(baseline_ecmwf_file_list).sel(L=leads).sel(S=slice(test_start,test_end)).astype(np.float32).load()
    
    
    #Need to open a template of ECMWF to mask the np.nan values that
    return(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, var_OUT, template_testing_only)



def open_obs_for_verification(region_name, leads,train_start, train_end, val_start, val_end, test_start, test_end,soil_dir):
    print('Loading all the baseline files for observations')
    
    obs_anomaly_SubX_format =xr.open_mfdataset(f'{soil_dir}/{region_name}/RZSM_anomaly_reformat_SubX_format/RZSM_anomaly*.nc4').sel(L=leads).astype(np.float32).load()
    
    train = obs_anomaly_SubX_format.sel(S=slice(train_start,train_end)).load()
    val = obs_anomaly_SubX_format.sel(S=slice(val_start,val_end)).load()
    test = obs_anomaly_SubX_format.sel(S=slice(test_start,test_end)).load()

    return(train, val, test)

def global_max_min(dictionary: dict, var_name: str) -> int:
    max_, min_ = [], []
    for k,v in dictionary.items():
        for region,r in dictionary[k].items():
            max_.append(np.nanmax(dictionary[k][var_name].values))
            min_.append(np.nanmin(dictionary[k][var_name].values))

    return(max(max_), min(min_))

#First randomly select a region and a grid cell number
def random_region_and_grid_selection(obs_anom_percentile_thresholds):
    random.seed(os.getpid())
    reg = np.random.choice(['CONUS','china','australia'])
    grid_x = np.random.choice(obs_anom_percentile_thresholds[reg].X.shape[0])
    grid_y = np.random.choice(obs_anom_percentile_thresholds[reg].Y.shape[0])

    if np.isnan(obs_anom_percentile_thresholds[reg]['20th_percentile'][100,grid_y,grid_x].values):
        proceed = 'No'
        return(proceed, reg, grid_x, grid_y)
    else:
        proceed = 'Yes'
        return(proceed, reg, grid_x, grid_y)



def open_all_files_for_sampling(obs_anomaly_SubX_format, baseline_anomaly, baseline_ecmwf, reg, grid_x, grid_y, day_num,obs_anom_percentile_thresholds, 
                                percentile,UNET_experiment_name,week_lead, test_year,obs_anom_percentile):
    
    # UNET_experiment_name = 'EX6_regular_RZSM'
    
    obs = obs_anomaly_SubX_format[reg][putils.xarray_varname(obs_anomaly_SubX_format[reg])].isel(X=grid_x, Y=grid_y).sel(L=day_num)
    gefs= baseline_anomaly[reg][putils.xarray_varname(baseline_anomaly[reg])].isel(X=grid_x, Y=grid_y).sel(L=day_num)
    ecmwf= baseline_ecmwf[reg][putils.xarray_varname(baseline_ecmwf[reg])].isel(X=grid_x, Y=grid_y).sel(L=day_num)

    obs_anom_percentile_thresholds_out = obs_anom_percentile_thresholds[reg][f'{percentile}th_percentile'].isel(X=grid_x, Y=grid_y)
    obs_anom_percentile_out = obs_anom_percentile[reg][f'{percentile}th_percentile'].isel(X=grid_x, Y=grid_y).sel(L=day_num)

    unet_files = sorted(glob(f'predictions/{reg}/Wk{week_lead}_testing/*{UNET_experiment_name}*')) #Will all data
    gef_unet = [i for i in unet_files if f'{UNET_experiment_name}' in i][0]
    test_name = gef_unet.split('testing_')[-1].split('.npy')[0]

    if 'ECMWF' in gef_unet:
        new_source = 'ECMWF'
    else:
        new_source = 'GEFSv12'

    add_to_file = load_UNET_files_sampling(gefs, gef_unet, day_num,new_source,test_year, grid_x, grid_y,test_name, region_name = reg)

    return(obs, gefs, ecmwf, obs_anom_percentile_thresholds_out, obs_anom_percentile_out, add_to_file)



def load_UNET_files_sampling(gefs, file, day_num,new_source,test_year, grid_x, grid_y,test_name,region_name):
    add_to_file = gefs.copy(deep = True)
    
    load_ = np.load(file)[2,:,:,:,0] 
    test = reverse_min_max_scaling(load_, region_name, day_num,new_source,test_year)
    
    test = np.reshape(test,(test.shape[0]//11,11,test.shape[1],test.shape[2]))
    test = np.expand_dims(test, -1)
    #Now re-order the dimensions to match SubX
    load_ =  np.reshape(test,(test.shape[0], test.shape[1], test.shape[-1], test.shape[2], test.shape[3]))
    load_.shape
    load_ = load_[:,:,0,grid_y, grid_x]
    add_to_file[:,:] = load_

    return(add_to_file)


def accuracy(obs_below_final, forecast, ensemble_threshold):
    TP = 0
    for S in obs_below_final.S.values:
        # break
        '''First compute only with observations where they are below the 20th percentile'''
        if ~np.isnan(obs_below_final.sel(S=S)):
            if (forecast.sel(S=S) >= ensemble_threshold*10):
                TP +=1

    '''Now see where the incorrect predictions were'''
    FP = 0
    for S in obs_below_final.S.values:
        # break
        '''First compute only with observations where they are below the 20th percentile'''
        if np.isnan(obs_below_final.sel(S=S)) and (forecast.sel(S=S) >= ensemble_threshold*10):
            FP +=1

    TN = 0
    for S in obs_below_final.S.values:
        # break
        '''First compute only with observations where they are below the 20th percentile'''
        if np.isnan(obs_below_final.sel(S=S)) and (forecast.sel(S=S) <= ensemble_threshold*10):
            TN +=1

    '''Now see where the incorrect predictions were'''
    FN = 0
    for S in obs_below_final.S.values:
        # break
        '''First compute only with observations where they are below the 20th percentile'''
        if ~np.isnan(obs_below_final.sel(S=S)) and (forecast.sel(S=S) <= ensemble_threshold*10):
            FN +=1    
            
    return(TP, FP, TN, FN, TP+FP+TN+FN)
            

def identify_location():
    return(print(os.getcwd()))


def compute_anomaly_on_testing(full_file, train_end, test_file):
    climatology_season = return_seasonal_mean(full_file,train_end)
    climatology_season = climatology_season.rename({'member':'M','lead':'L','lat':'Y','lon':'X'})
    
    summer_= test_file.sel(S=(test_file['S.season']=='JJA')) - climatology_season.sel(season='JJA')
    fall_= test_file.sel(S=(test_file['S.season']=='SON'))- climatology_season.sel(season='SON')
    winter_= test_file.sel(S=(test_file['S.season']=='DJF'))- climatology_season.sel(season='DJF')
    spring_= test_file.sel(S=(test_file['S.season']=='MAM'))- climatology_season.sel(season='MAM')

    combined_files = xr.concat([summer_,fall_,winter_,spring_],dim='S').sortby('S')
    combined_files = combined_files.drop('season')
    return(combined_files)

def return_seasonal_mean(file,train_end):
    climatology_season = file.sel(init=(file['init.year'] <= train_end)).groupby(f"init.{seasonality_str}").mean()
    return(climatology_season)

def compute_std(forecast):
    '''All seasons'''
    #First find the standard deviation of the spread
    ensemble_spread = forecast.std(dim='member')  # shape: (init, lead, lat, lon)
    climatological_spread = ensemble_spread.mean(dim='init')  # shape: (lead, lat, lon)
    return climatological_spread


    