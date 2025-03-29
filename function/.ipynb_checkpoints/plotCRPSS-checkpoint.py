#!/usr/bin/env python3
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from glob import glob

def return_clim_persistence_and_initialized(file):
    # Extract CRPSS components
    persist = file['crpss'].sel(skill='persistence', results='verify skill').drop_vars(['skill','results'])
    initialized = file['crpss'].sel(skill='initialized', results='verify skill').drop_vars(['skill','results'])
    climatology = file['crpss'].sel(skill='climatology', results='verify skill').drop_vars(['skill','results'])
    return persist, initialized, climatology

def load_bootstrap_file_all_season(region_name, obs_source, forecast_name):
    file_path = f'Data/crpss_bootstrap/{region_name}/all_season_skill_{forecast_name}_forecast_{obs_source}_obs.nc'
    with xr.open_dataset(file_path) as ds:
        return ds.load()  # load into memory, closes file after context block

def load_bootstrap_file_by_season(region_name, obs_source, forecast_name, season):
    file_path = f'Data/crpss_bootstrap/{region_name}/{season}_skill_{forecast_name}_forecast_{obs_source}_obs.nc'
    with xr.open_dataset(file_path) as ds:
        return ds.load()  # load into memory, closes file after context block

def return_initialized(file):
    # Extract CRPSS components
    initialized = file['crpss'].sel(skill='initialized', results='verify skill').drop_vars(['skill','results'])
    return initialized


def plot_CRPSS_components(region_name, obs_source, forecast_name):
    """Plot CRPSS which is greater than 0. This shows that it is better than climatology."""

    # Output directory
    new_dir = f'Outputs/CRPSS_bootstrap/crpss_components/{region_name}'
    os.makedirs(new_dir, exist_ok=True)

    # Load the bootstrap file
    file_bc = load_bootstrap_file_all_season(region_name, obs_source, f'{forecast_name}-BC')
    unet = load_bootstrap_file_all_season(region_name, obs_source, f'EX29_{forecast_name}')

    
    #bias-corrected
    Bpersist, Binitialized, Bclimatology = return_clim_persistence_and_initialized(file_bc)
    
    #Unet
    Upersist, Uinitialized, Uclimatology = return_clim_persistence_and_initialized(unet)
    
    leads = [0, 1, 2, 3]
    
    if forecast_name=='ECMWF':
        name_ = 'DL-DM_E'
    else:
        name_ = 'DL-DM_G'
        
    
    # all_data = xr.concat([Binitialized, Uinitialized], dim='lead')
    # all_data = xr.where(all_data >= 0, all_data, np.nan)
    all_data = [Binitialized, Uinitialized]
    all_data_names = [f'{forecast_name}-BC', name_]
    
    lon = all_data[0].lon.values
    lat = all_data[0].lat.values
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    
    
    fig, axs = plt.subplots(
        4, 2,  # 4 leads (rows), 3 columns (climatology, init)
        figsize=(15, 10),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    max1,max2 = all_data[0].mean(dim='init').max().values, all_data[1].mean(dim='init').max().values
    
    #Only keeping values if they are greater than climatology
    vmin = 0
    vmax = max(max1,max2)
    cmap = 'YlOrRd'
    
    # Normalize color scales across all subplots
    norm_top = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    
    for row, lead in enumerate(leads):
        for col, data_type in enumerate(all_data_names):
            ax = axs[row, col]
            
            data = all_data[col].isel(lead=lead).mean(dim='init')
    
            data = xr.where(data >= 0, data, np.nan) #This would keep only values greater than 0 which are better than climatology
    
            im = ax.pcolormesh(mesh_lon, mesh_lat, data, cmap=cmap, norm=norm_top,
                               transform=ccrs.PlateCarree(), shading='auto')
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)


            if row == 0:
                ax.set_title(all_data_names[col], fontsize=12)
    
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
            gl.top_labels = gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            
            if col == 0:
                ax.text(-0.15, 0.5, f'Week {lead + 1}', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right')  
            
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(norm=norm_top, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('CRPSS', fontsize=12)
    
    plt.suptitle(f"Initialized Forecast > Climatology\nFor {obs_source} Reanalysis", fontsize=16, y=0.94)
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])

    save_file = f'{new_dir}/CRPSS_greater_than_climatology_{forecast_name}_{obs_source}.png'
    print(f'Saving image to {save_file}')
    plt.savefig(save_file, dpi=300)
    plt.show()


def plot_CRPSS_components_combined(region_name, obs_source):
    """Plot CRPSS which is greater than 0. This shows that it is better than climatology."""

    # Output directory
    new_dir = f'Outputs/CRPSS_bootstrap/crpss_components/{region_name}'
    os.makedirs(new_dir, exist_ok=True)

    # Load the bootstrap file
    file_ecmwf = load_bootstrap_file_all_season(region_name, obs_source, f'ECMWF-BC')
    file_gefs = load_bootstrap_file_all_season(region_name, obs_source, f'GEFSv12-BC')
    unet_ecmwf = load_bootstrap_file_all_season(region_name, obs_source, f'EX29_ECMWF')
    unet_gefs = load_bootstrap_file_all_season(region_name, obs_source, f'EX29_GEFSv12')

    
    #bias-corrected GEFS
    Gpersist, Ginitialized, Gclimatology = return_clim_persistence_and_initialized(file_gefs)
    Epersist, Einitialized, Eclimatology = return_clim_persistence_and_initialized(file_ecmwf)
    
    #Unet
    EUpersist, EUinitialized, EUclimatology = return_clim_persistence_and_initialized(unet_ecmwf)
    GUpersist, GUinitialized, GUclimatology = return_clim_persistence_and_initialized(unet_gefs)
    
    leads = [0, 1, 2, 3]
            
    
    # all_data = xr.concat([Binitialized, Uinitialized], dim='lead')
    # all_data = xr.where(all_data >= 0, all_data, np.nan)
    all_data = [Ginitialized, GUinitialized,Einitialized, EUinitialized]
    all_data_names = [f'GEFSV12-BC', 'DL-DM_G','ECMWF-BC','DL-DM_E']
    
    lon = all_data[0].lon.values
    lat = all_data[0].lat.values
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    
    

    fig, axs = plt.subplots(
        4, 4,  # 4 leads (rows), 4 columns 
        figsize=(15, 10),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    
    max_ = [all_data[i].mean(dim='init').max().values for i in range(len(all_data))]
    min_ = [all_data[i].mean(dim='init').min().values for i in range(len(all_data))]
    
    #Only keeping values if they are greater than climatology
    vmin = 0
    vmax = max(max_)
    cmap = 'YlOrRd'
    
    # Normalize color scales across all subplots
    norm_top = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    
    for row, lead in enumerate(leads):
        for col, data_type in enumerate(all_data_names):
            ax = axs[row, col]
            
            data = all_data[col].isel(lead=lead).mean(dim='init')
    
            data = xr.where(data >= 0, data, np.nan) #This would keep only values greater than 0 which are better than climatology
    
            im = ax.pcolormesh(mesh_lon, mesh_lat, data, cmap=cmap, norm=norm_top,
                               transform=ccrs.PlateCarree(), shading='auto')
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(f'Lead: {lead+1} week', fontsize=10)
    
            if row == 0:
                ax.set_title(all_data_names[col], fontsize=12)
    
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
            gl.top_labels = gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            gl.left_labels = True  # <-- add this
            if col == 0:
                ax.text(-0.15, 0.5, f'Week {lead + 1}', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right')  
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(norm=norm_top, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('CRPSS', fontsize=12)
    
    plt.suptitle(f"Initialized Forecast > Climatology\n{obs_source} Reanalysis as Observation", fontsize=16, y=0.94)
    # plt.tight_layout(rect=[0.05, 0, 0.9, 1])  # Increase left margin (first number)
    
    save_file = f'{new_dir}/CRPSS_greater_than_climatology_combined_{obs_source}.png'
    print(f'Saving image to {save_file}')
    plt.savefig(save_file, dpi=300)
    plt.show()


def plot_CRPSS_components_seasons_by_week(region_name, obs_source,week_lead):
    """Plot CRPSS which is greater than 0. This shows that it is better than climatology."""

    # Output directory
    new_dir = f'Outputs/CRPSS_bootstrap/crpss_components/{region_name}/seasonal'
    os.makedirs(new_dir, exist_ok=True)


    seasons = ['DJF','MAM','JJA','SON']
    
    # Load the bootstrap file
    file_ecmwf = {season:load_bootstrap_file_by_season(region_name, obs_source, f'ECMWF-BC',season).isel(lead=week_lead -1) for season in seasons}
    file_gefs = {season:load_bootstrap_file_by_season(region_name, obs_source, f'GEFSv12-BC',season).isel(lead=week_lead -1)  for season in seasons}
    # Load the bootstrap file
    unet_ecmwf = {season:load_bootstrap_file_by_season(region_name, obs_source, f'EX29_ECMWF',season).isel(lead=week_lead -1)  for season in seasons}
    unet_gefs = {season:load_bootstrap_file_by_season(region_name, obs_source, f'EX29_GEFSv12',season).isel(lead=week_lead -1)  for season in seasons}


    #bias-corrected GEFS
    Ginitialized  = {season:return_initialized(file_gefs[season]) for season in seasons}
    #bias-corrected ECMWF
    Einitialized = {season:return_initialized(file_ecmwf[season]) for season in seasons}
    
    #Unet ECMWF
    EUinitialized = {season:return_initialized(unet_ecmwf[season]) for season in seasons}
    #Unet GEFSv12
    GUinitialized = {season:return_initialized(unet_gefs[season]) for season in seasons}
        

    m1,m2,m3,m4 = max([Ginitialized[season].mean(dim='init').max().values for season in seasons]),\
    max([Einitialized[season].mean(dim='init').max().values for season in seasons]),\
    max([EUinitialized[season].mean(dim='init').max().values for season in seasons]),\
    max([GUinitialized[season].mean(dim='init').max().values for season in seasons])
    
    all_data = [Ginitialized, GUinitialized, Einitialized, EUinitialized]
    all_data_names = ['GEFSv12-BC', 'DL-DM_G','ECMWF-BC','DL-DM_E']
    
    lon = all_data[0]['DJF'].lon.values
    lat = all_data[0]['DJF'].lat.values
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    
    
    fig, axs = plt.subplots(
        4, 4,  # 4 leads (rows), 4 columns 
        figsize=(15, 10),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    #Only keeping values if they are greater than climatology
    vmin = 0
    vmax = np.nanmax(np.array([m1,m2,m3,m4]))
    cmap = 'YlOrRd'
    
    # Normalize color scales across all subplots
    norm_top = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    
    for row, season in enumerate(seasons):
        for col, data_type in enumerate(all_data_names):
            ax = axs[row, col]
            
            data = all_data[col][season].mean(dim='init')
    
            data = xr.where(data >= 0, data, np.nan) #This would keep only values greater than 0 which are better than climatology
    
            im = ax.pcolormesh(mesh_lon, mesh_lat, data, cmap=cmap, norm=norm_top,
                               transform=ccrs.PlateCarree(), shading='auto')
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
            if row == 0:
                ax.set_title(all_data_names[col], fontsize=12)
    
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
            gl.top_labels = gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            gl.left_labels = True  # <-- add this
            if col == 0:
                ax.text(-0.15, 0.5, season, transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right')  
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(norm=norm_top, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('CRPSS', fontsize=12)
    
    plt.suptitle(f"Initialized Forecast > Climatology\n{obs_source} Reanalysis as Observation\nWeek {week_lead}", fontsize=16, y=0.97)
    # plt.tight_layout(rect=[0.05, 0, 0.9, 1])  # Increase left margin (first number)
    
    save_file = f'{new_dir}/Wk{week_lead}_seasonal_CRPSS_greater_than_climatology_combined_{obs_source}.png'
    print(f'Saving file as {save_file}')
    plt.savefig(save_file, dpi=300)
    plt.show()
    
    