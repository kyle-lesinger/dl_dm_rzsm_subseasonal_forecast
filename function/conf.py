#!/user/bin/env python3


#Setup where data/home environment is setup. This is a root/base so very important!
home = '/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning'


global gefsv12_data, ecmwf_data
#reforecast products
gefsv12_data = f'{home}/Data/reforecast/GEFSv12'
ecmwf_data = f'{home}/Data/reforecast/ECMWF'

global gleam_data, era_data, base_reanalysis
#reanalysis products
gleam_data = f'{home}/Data/reanalysis/GLEAM'
era_data = f'{home}/Data/reanalysis/ERA5'
base_reanalysis = f'{home}/Data/reanalysis'


def return_data_directories(reforecast_input, region_name):
    #Gleam observations
    gleam_dir = f'{gleam_data}/{region_name}'
    
    #ERA5 observations
    era5_dir = f'{era_data}/{region_name}'
    
    
    #Either GEFSv12 or ECMWF
    if reforecast_input == 'ECMWF':
        fcst_dir = f'{ecmwf_data}/{region_name}'
    elif reforecast_input == 'GEFSv12': 
        fcst_dir = f'{gefsv12_data}/{region_name}'

    return gleam_dir, era5_dir, fcst_dir, base_reanalysis



dim_order = ['S','M','L','Y','X']
