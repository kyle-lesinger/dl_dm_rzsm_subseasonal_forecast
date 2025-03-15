#!/bin/bash

module load cdo


out_australia=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/ERA5/australia

mask_australia=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/masks/australia_0.5_grid.grd


cdo remapbil,"$mask_australia" swvl_total.nc "$out_australia/RZSM_weighted_mean_0_100cm.nc4"
