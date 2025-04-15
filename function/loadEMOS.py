#!/usr/bin/env python3
from glob import glob
import xarray as xr
from function import verifications
import numpy as np


def return_EMOS_average(region_name,test_start,test_end,obs_original,df_acc, df_crps,ecmwf_acc):
    print('Adding EMOS results')
    '''This is for adding all 4 EMOS experiments, but for some ready it is breaking right now and I'm not sure why (it kills the kernel)'''
    emos_files_full = sorted(glob(f'Data/EMOS/{region_name}/EMOS_11*test_predictions*.nc'))
    
    # Loop through all the EMOS file experiments
    e_acc, e_crps = ecmwf_acc.copy(deep=True), ecmwf_acc.copy(deep=True)

    e_acc[putils.xarray_varname(e_acc)][:,:,:],  e_crps[putils.xarray_varname(e_crps)][:,:,:] = 0, 0

    # Loop through all the EMOS file experiments
    for idx,file in enumerate(emos_files_full):
        # break
        emos_ = xr.open_dataset(file).rename({'idate':'S', 'model': 'M','vdate': 'L', 'latitude': 'Y', 'longitude': 'X'}).sel(S=slice(test_start,test_end))
        #First get the ACC values of GEFS and ECMWF relative to observations
        emos_acc = verifications.create_climpred_ACC(verifications.rename_subx_for_climpred(emos_), verifications.rename_obs_for_climpred(obs_original))
        add_  = (e_acc[putils.xarray_varname(e_acc)].values + emos_acc[putils.xarray_varname(emos_acc)].sel(lead=[6,13,20,27]).values)
        e_acc[putils.xarray_varname(e_acc)][:,:,:] = add_

        emos_crps = verifications.create_climpred_CRPS(verifications.rename_subx_for_climpred(emos_), verifications.rename_obs_for_climpred(obs_original))
        emos_crps = emos_crps.mean(dim='init')
        add_  = (e_crps[putils.xarray_varname(e_crps)].values + emos_crps[putils.xarray_varname(emos_crps)].sel(lead=[6,13,20,27]).values)
        e_crps[putils.xarray_varname(e_crps)][:,:,:] = add_

    #Divide
    e_acc = e_acc /len(emos_files_full)
    e_crps = e_crps /len(emos_files_full)
    #RE ADD MASK
    mm = ~np.isnan(emos_.isel(M=0,S=0).sel(L=[6,13,20,27]))[putils.xarray_varname(emos_)].values
    e_acc = xr.where(mm==True,e_acc,np.nan)
    e_crps = xr.where(mm==True,e_crps,np.nan)
    
    df_acc = add_lineplot_to_dataframe(df_acc,e_acc,'EMOS', 'ACC',10,'mean')
    df_crps = add_lineplot_to_dataframe(df_crps,e_crps,'EMOS', 'CRPS',10,'median')
    return df_acc, df_crps