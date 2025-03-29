#!/usr/bin/env python3
import xarray as xr
import os
from . import preprocessUtils as putils
from . import conf

global dir
dir = conf.home

def load_mask(region_name):
    #Load contiguous united states mask. This can be any arbitrary mask for files
    return(xr.open_dataset(f'{dir}/Data/masks/{region_name}_mask.nc4'))


# def load_mask(region_name):
#     #Load contiguous united states mask. This can be any arbitrary mask for files
#     if region_name == 'CONUS':
#         return(xr.open_dataset(f'{dir}/Data/CONUS_mask/CONUS_mask.nc'))
#     elif region_name == 'australia':
#         return(xr.open_dataset(f'{dir}/Data_australia/GLEAM/australia_mask.nc4'))
#     elif region_name == 'china':
#         return(xr.open_dataset(f'{dir}/Data_china/GLEAM/china_mask.nc4'))

def load_region_mask(region_name):
    #Load contiguous united states mask. This can be any arbitrary mask for files
    if region_name == 'CONUS':
        correct_shape = xr.open_dataset(f'{dir}/Data/masks/CONUS_mask.nc4')
        region_mask = xr.open_dataset(f'{dir}/Data/masks/region_CONUS_mask.nc4')
        #Now select the correct dimensions
        return(region_mask.sel(latitude=correct_shape.Y.values,longitude=correct_shape.X.values)['NCAregions_mask'])
    elif region_name == 'australia':
        return(xr.open_dataset(f'{dir}/Data_australia/GLEAM/australia_mask.nc4'))


def load_mask_vals(region_name):
    mask = load_mask(region_name)
    #Mask with np.nan for non-CONUS land values
    try:
        mask_anom = mask[putils.xarray_varname(mask)][0,:,:].values
    except IndexError:
        mask_anom = mask[putils.xarray_varname(mask)][:,:].values
    return(mask,mask_anom)