#!/usr/bin/env python3

import xarray as xr

def load_additive_bias_corrected_data_ACC(leads,region_name,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC = xr.open_dataset(f'{data_dir}/ecmwf_acc_values_{obs_source}.nc').isel(lead=leads)
    gef_BC = xr.open_dataset(f'{data_dir}/gefs_acc_values_{obs_source}.nc').isel(lead=leads)
    ecm_BC['lead'] = leads
    gef_BC['lead'] = leads
    return gef_BC, ecm_BC

def load_additive_bias_corrected_data_CRPS(leads,region_name,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC = xr.open_dataset(f'{data_dir}/ecmwf_crps_values_{obs_source}.nc').isel(lead=leads)
    gef_BC = xr.open_dataset(f'{data_dir}/gefs_crps_values_{obs_source}.nc').isel(lead=leads)
    ecm_BC['lead'] = leads
    gef_BC['lead'] = leads
    return gef_BC, ecm_BC

def load_additive_bias_anomaly(leads,region_name,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC = xr.open_dataset(f'{data_dir}/ecmwf_anomaly_values_bias_corrected_testing_years_{obs_source}.nc').isel(lead=leads)
    gef_BC = xr.open_dataset(f'{data_dir}/gefs_anomaly_values_bias_corrected_testing_years_{obs_source}.nc').isel(lead=leads)
    ecm_BC['lead'] = leads
    gef_BC['lead'] = leads
    return gef_BC, ecm_BC

def load_additive_bias_corrected_data_by_season(leads,region_name,metric,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC_djf = xr.open_dataset(f'{data_dir}/ecmwf_{metric}_values_DJF_{obs_source}.nc').isel(lead=leads)
    gef_BC_djf = xr.open_dataset(f'{data_dir}/gefs_{metric}_values_DJF_{obs_source}.nc').isel(lead=leads)
    ecm_BC_djf['lead'] = leads
    gef_BC_djf['lead'] = leads

    ecm_BC_mam = xr.open_dataset(f'{data_dir}/ecmwf_{metric}_values_MAM_{obs_source}.nc').isel(lead=leads)
    gef_BC_mam = xr.open_dataset(f'{data_dir}/gefs_{metric}_values_MAM_{obs_source}.nc').isel(lead=leads)
    ecm_BC_mam['lead'] = leads
    gef_BC_mam['lead'] = leads

    ecm_BC_jja = xr.open_dataset(f'{data_dir}/ecmwf_{metric}_values_JJA_{obs_source}.nc').isel(lead=leads)
    gef_BC_jja = xr.open_dataset(f'{data_dir}/gefs_{metric}_values_JJA_{obs_source}.nc').isel(lead=leads)
    ecm_BC_jja['lead'] = leads
    gef_BC_jja['lead'] = leads

    ecm_BC_son = xr.open_dataset(f'{data_dir}/ecmwf_{metric}_values_SON_{obs_source}.nc').isel(lead=leads)
    gef_BC_son = xr.open_dataset(f'{data_dir}/gefs_{metric}_values_SON_{obs_source}.nc').isel(lead=leads)
    ecm_BC_son['lead'] = leads
    gef_BC_son['lead'] = leads
    
    return ecm_BC_djf, gef_BC_djf, ecm_BC_mam, gef_BC_mam, ecm_BC_jja, gef_BC_jja, ecm_BC_son, gef_BC_son

def load_additive_bias_corrected_data_CRPSS(leads,region_name,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC = xr.open_dataset(f'{data_dir}/ecmwf_crpss_values_{obs_source}.nc').isel(lead=leads)
    gef_BC = xr.open_dataset(f'{data_dir}/gefs_crpss_values_{obs_source}.nc').isel(lead=leads)
    ecm_BC['lead'] = leads
    gef_BC['lead'] = leads
    return gef_BC, ecm_BC

def load_additive_bias_corrected_data_CRPSS_season(leads,region_name,obs_source):
    '''This was already calculated in a previous script'''
    data_dir = f'Data/bias_corrected_reforecast/{region_name}'
    ecm_BC_djf = xr.open_dataset(f'{data_dir}/ecmwf_crpss_values_DJF_{obs_source}.nc').isel(lead=leads)
    gef_BC_djf = xr.open_dataset(f'{data_dir}/gefs_crpss_values_DJF_{obs_source}.nc').isel(lead=leads)
    ecm_BC_djf['lead'] = leads
    gef_BC_djf['lead'] = leads

    ecm_BC_mam = xr.open_dataset(f'{data_dir}/ecmwf_crpss_values_MAM_{obs_source}.nc').isel(lead=leads)
    gef_BC_mam = xr.open_dataset(f'{data_dir}/gefs_crpss_values_MAM_{obs_source}.nc').isel(lead=leads)
    ecm_BC_mam['lead'] = leads
    gef_BC_mam['lead'] = leads

    ecm_BC_jja = xr.open_dataset(f'{data_dir}/ecmwf_crpss_values_JJA_{obs_source}.nc').isel(lead=leads)
    gef_BC_jja = xr.open_dataset(f'{data_dir}/gefs_crpss_values_JJA_{obs_source}.nc').isel(lead=leads)
    ecm_BC_jja['lead'] = leads
    gef_BC_jja['lead'] = leads

    ecm_BC_son = xr.open_dataset(f'{data_dir}/ecmwf_crpss_values_SON_{obs_source}.nc').isel(lead=leads)
    gef_BC_son = xr.open_dataset(f'{data_dir}/gefs_crpss_values_SON_{obs_source}.nc').isel(lead=leads)
    ecm_BC_son['lead'] = leads
    gef_BC_son['lead'] = leads
    
    return ecm_BC_djf, gef_BC_djf, ecm_BC_mam, gef_BC_mam, ecm_BC_jja, gef_BC_jja, ecm_BC_son, gef_BC_son