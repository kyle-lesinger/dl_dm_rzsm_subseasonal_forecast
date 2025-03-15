#!/usr/bin

module load cdo

#Remap to a 0.5 grid
grid_5="remap_0.5.grd"
rem=GLEAM_remap
mkdir -p $rem

# Loop through all NetCDF files
for file in *.nc; do
    echo "Processing: $file"
    
    output_file=$rem/$file

    # Check if output file already exists
    if [[ ! -f "$output_file" ]]; then
        cdo remapbil,"$grid_5" "$file" "$output_file"
    else
        echo "Skipping: $file (already processed)"
    fi
done

merge=$rem/merge_by_type
mkdir -p $merge
#Merge files of the same type
cdo mergetime $rem/SMrz* $merge/SMroot_all.nc
cdo mergetime $rem/SMs* $merge/SMsurface_all.nc

#Then weight each one and add them
cdo add -mulc,0.9 $merge/SMroot_all.nc -mulc,0.1 $merge/SMsurface_all.nc $merge/RZSM_weighted_mean_0_100cm.nc4



# masks of each region
mask_dir=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/masks

conus=$mask_dir/conus_0.5_grid.grd
china=$mask_dir/china_0.5_grid.grd
australia=$mask_dir/australia_0.5_grid.grd

cdo remapbil,$australia $merge/RZSM_weighted_mean_0_100cm.nc4 $merge/australia.nc4
cdo remapbil,$china $merge/RZSM_weighted_mean_0_100cm.nc4 $merge/china.nc4
cdo remapbil,$conus $merge/RZSM_weighted_mean_0_100cm.nc4 $merge/conus.nc4


cp $merge/china.nc4 /glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/GLEAM/china/RZSM_weighted_mean_0_100cm.nc4
cp $merge/australia.nc4 /glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/GLEAM/australia/RZSM_weighted_mean_0_100cm.nc4
cp $merge/conus.nc4 /glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/reanalysis/GLEAM/CONUS/RZSM_weighted_mean_0_100cm.nc4