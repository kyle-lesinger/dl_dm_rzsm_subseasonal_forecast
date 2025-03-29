#!/usr/bin/env python3

import numpy as np

def load_min_max_RZSM(region_name, lead, obs_or_forecast):
    min_max_dir = f'Data/min_max_values/{region_name}'

    source = 'GLEAM' if  obs_or_forecast == 'obs' else 'GEFSv12'
    
    max_RZSM_OBS = np.load(f'{min_max_dir}/soilw_bgrnd_{source}_max.npy')
    min_RZSM_OBS = np.load(f'{min_max_dir}/soilw_bgrnd_{source}_min.npy')
    
    return(max_RZSM, min_RZSM)

def load_observation_min_max_RZSM_grid_cell_standardization(region_name):
    min_max_dir = f'Data/min_max_values/{region_name}'
    max_RZSM_OBS = np.load(f'{min_max_dir}/soilw_bgrnd_GLEAM_max_by_grid_cell.npy')
    min_RZSM_OBS = np.load(f'{min_max_dir}/soilw_bgrnd_GLEAM_min_by_grid_cell.npy')
    return(max_RZSM_OBS, min_RZSM_OBS)

def load_observation_min_max_TMAX(region_name):
    min_max_dir = f'Data/min_max_values/{region_name}'
    max_tmax_OBS = np.load(f'{min_max_dir}/tmax_ERA5_max.npy')
    min_tmax_OBS = np.load(f'{min_max_dir}/tmax_ERA5_min.npy')
    return(max_tmax_OBS, min_tmax_OBS)

def load_reforecast_min_max_RZSM(region_name):
    min_max_dir = f'Data/min_max_values/{region_name}'
    max_RZSM_reforecast = np.load(f'{min_max_dir}/soilw_bgrnd_GEFSv12_max.npy')
    min_RZSM_reforecast = np.load(f'{min_max_dir}/soilw_bgrnd_GEFSv12_min.npy')
    return(max_RZSM_reforecast, min_RZSM_reforecast)

def load_reforecast_min_max_RZSM_grid_cell_standardization(region_name):
    min_max_dir = f'Data/min_max_values/{region_name}'
    max_RZSM_reforecast = np.load(f'{min_max_dir}/soilw_bgrnd_GEFSv12_max_by_grid_cell.npy')
    min_RZSM_reforecast = np.load(f'{min_max_dir}/soilw_bgrnd_GEFSv12_min_by_grid_cell.npy')
    return(max_RZSM_reforecast, min_RZSM_reforecast)



def load_reforecast_min_max_TMAX(region_name):
    min_max_dir = f'Data/min_max_values/{region_name}'
    max_tmax_reforecast = np.load(f'{min_max_dir}/tmax_GEFSv12_max.npy')
    min_tmax_reforecast = np.load(f'{min_max_dir}/tmax_GEFSv12_min.npy')
    return(max_tmax_reforecast, min_tmax_reforecast)