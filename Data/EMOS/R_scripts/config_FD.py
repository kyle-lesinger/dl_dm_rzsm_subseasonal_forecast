#!/usr/bin/env python3

#Where reanalysis GEFSv12 is located
reanalysis_media_dir = '/media/kdl/DONT_KNOW/GEFSv12_reforecast_raw_0.5degree_CONUS/GEFSv12'

#Home directory. Where most of the scripts will begin and data will be saved
home_dir='/home/kdl/Insync/OneDrive/flashDrought_DL_project/'

#number of processors for multiprocessing functions
n_processors=8

#GLEAM observations directory
gleam_dir=f'{home_dir}/Data/GLEAM_v3.7a'

#CONUS mask directory
conus_dir=f'{home_dir}/Data/CONUS_masks_grids'

#EMC reforecast (SubX directory)
subx_dir=f'{home_dir}/Data/EMC_reforecast'

#EMOS directory
emos_dir=f'{home_dir}/Data/EMOS'

#UNET input data (that is sent to HPC for modeling)
unet_input_dir=f'{home_dir}/Data/model_inputs_renalysis_reforecast'

