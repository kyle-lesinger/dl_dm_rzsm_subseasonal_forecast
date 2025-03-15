#!/bin/bash

module load cdo


out_china=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/ERA5/china

mask_china=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/masks/china_0.5_grid.grd

cdo remapbil,"$mask_china" swvl_total.nc "$out_china/RZSM_weighted_mean_0_100cm.nc4"

