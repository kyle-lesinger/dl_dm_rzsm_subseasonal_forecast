#!/bin/bash

module load cdo

out_conus=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/ERA5/CONUS

mask_conus=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/masks/conus_0.5_grid.grd

cdo remapbil,"$mask_conus" swvl_total.nc "$out_conus/RZSM_weighted_mean_0_100cm.nc4"