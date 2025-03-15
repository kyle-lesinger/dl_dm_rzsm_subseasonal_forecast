#!/bin/bash

module load cdo

home_dir=pwd/RAW_ERA5_china

#DAYMEAN ONLY
vars("2m_dewpoint_temperature" "2m_temperature" "geopotential_height" "surface_pressure" "total_column_water")
for var in "${names[@]}";do

cd $home_dir/$var 

dir=daymean

mkdir  $home_dir/$var/$dir

for file in *.nc;do

if ! test -f  $home_dir/$var/$dir/$file; then

cdo daymean $file  $home_dir/$var/$dir/$file
fi;
done


# "maximum_2m_temperature" "minimum_2m_temperature"


#Now merge files to a common directory
# out_dir=/glade/work/klesinger/FD_RZSM_deep_learning/Data_china/ERA5

# for var in "${names[@]}";do
# cdo mergetime $home_dir/$var 