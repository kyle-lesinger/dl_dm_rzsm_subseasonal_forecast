#!/bin/bash
module load conda
conda info --envs
conda activate /glade/work/klesinger/conda-envs/tf212gpu
module load ncl
module load cdo

# export NCARG_ROOT=/usr/local

#the purpose of this script is to a.) create download scripts for GEFSv12 and b.) subset by region 

# Maximum number of concurrent background processes
max_processes=25

homeDir=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data

maskDir=$homeDir/masks

aus_grid=$maskDir/australia_0.5_grid.grd
conus_grid=$maskDir/conus_0.5_grid.grd
china_grid=$maskDir/china_0.5_grid.grd
# First create wget scripts 

#Argv 1 is the script directory where files are to be saved
script=wget_scripts

#Argv 2 is the directory where raw files will be downloaded
save=/glade/derecho/scratch/klesinger/FD_RZSM_deep_learning/Data/raw_downloads/GEFSv12

mkdir $save $script

#You will need to manually add the variables that you want to download within the create_wget_scripts.py file (they are in a list)
# python3 create_wget_scripts.py $script $save

#Download the data
download_data (){
for file in $script/*;do
bash $file;
done
}

### To run the function to create wget scripts, uncomment the line below

# download_data

process_hgt_pressure () {

# Counter for tracking the number of concurrent background processes
counter=0

max_processes_=$2

#Now process hgt_pres individually
process_aus=$save/hgt_pres/australia_processed
process_conus=$save/hgt_pres/CONUS_processed
process_china=$save/hgt_pres/china_processed

mkdir $process_aus $process_conus $process_china

cd $save/hgt_pres

pressure_level=$1

for file in *.grib2; do

    process_file_ () {
    file_name="${file%??????}" #Must remove the last 6 characters in string
    
    final_out_conus=$process_conus/"${file_name}".nc
    final_out_australia=$process_aus/"${file_name}".nc
    final_out_china=$process_china/"${file_name}".nc

    #australia only 
    if test -f $final_out_australia; then
        echo "File already exists"
    else
        echo "Converting hgt_pres for Australia into $final_out_australia" 
        ncl_convert2nc $file -L
        #Now use cdo to convert to specific coordinates
        cdo -b f32 remapbil,$aus_grid -sellevel,"${pressure_level}" "${file_name}".nc $final_out_australia
        rm "${file_name}".nc
    fi


    #conus only 
    if test -f $final_out_conus; then
        echo "File already exists"
    else
        echo "Converting hgt_pres for CONUS into $final_out_conus" 
        ncl_convert2nc $file -L
        #Now use cdo to convert to specific coordinates
        cdo -b f32 remapbil,$conus_grid -sellevel,"${pressure_level}" "${file_name}".nc $final_out_conus
        rm "${file_name}".nc
    fi
    
    #china only
    if test -f $final_out_china; then
        echo "File already exists"
    else
        echo "Converting hgt_pres for China into $final_out_china" 
        ncl_convert2nc $file -L
        #Now use cdo to convert to specific coordinates
        cdo -b f32 remapbil,$china_grid -sellevel,"${pressure_level}" "${file_name}".nc $final_out_china
        rm "${file_name}".nc
    fi
    }

    process_file_ & 
    
    ((counter++))
    
    if [ "$counter" -ge "$max_processes_" ]; then
        # Wait for all background processes to finish before continuing
        wait
        # Reset the counter
        counter=0
    fi    
done

# Wait for the last batch of background processes to finish
wait
}


##################################################################################################################

process_other_variables () {


# Counter for tracking the number of concurrent background processes
counter=0

max_processes_=$2

#Now process hgt_pres individually
process_aus=$save/$1/australia_processed
process_conus=$save/$1/CONUS_processed
process_china=$save/$1/china_processed

mkdir $process_aus $process_conus $process_china

cd $save/$1

for file in *grib2; do

    process_file_ () {
    file_name="${file%??????}" #Must remove the last 6 characters in string
    
    final_out_conus=$process_conus/"${file_name}".nc
    final_out_australia=$process_aus/"${file_name}".nc
    final_out_china=$process_china/"${file_name}".nc


    # #Australia only
    # if test -f $final_out_australia; then
    #     echo "File already exists"
    # else
    #     echo "Converting $1 for Australia into $final_out_australia" 
        
    #     ncl_convert2nc $file -L
    #     #Now use cdo to convert to specific coordinates
    #     cdo -b f32 remapbil,$aus_grid "${file_name}".nc $final_out_australia
        
    #     rm "${file_name}".nc
    # fi

    #CONUS only
    if test -f $final_out_conus; then
        echo "File already exists"
    else
        echo "Converting $1 for CONUS into $final_out_conus" 
        
        ncl_convert2nc $file -L
        #Now use cdo to convert to specific coordinates
        cdo -b f32 remapbil,$conus_grid "${file_name}".nc $final_out_conus
        
        rm "${file_name}".nc
    fi

    # #china only
    # if test -f $final_out_china; then
    #     echo "File already exists"
    # else
    #     echo "Converting $1 for china into $final_out_china" 
        
    #     ncl_convert2nc $file -L
    #     #Now use cdo to convert to specific coordinates
    #     cdo -b f32 remapbil,$china_grid "${file_name}".nc $final_out_china
        
    #     rm "${file_name}".nc
    # fi
    }

    process_file_ & 
    
    ((counter++))
    
    if [ "$counter" -ge "$max_processes_" ]; then
        # Wait for all background processes to finish before continuing
        wait
        # Reset the counter
        counter=0
    fi    
done

wait
}

# #################################################################################################################


process_hgt_pressure "20000" $max_processes

process_other_variables "soilw_bgrnd" $max_processes
process_other_variables "tmax_2m" $max_processes
process_other_variables "tmin_2m" $max_processes
process_other_variables "tmp_2m" $max_processes
process_other_variables "spfh_2m" $max_processes
process_other_variables "pwat_eatm" $max_processes



# Now merge the files together (just seperate NH and australia right now). They have different size dimensions of the file
merge_files_together () {

var=$1
save_dir=/glade/work/klesinger/FD_RZSM_deep_learning/Data/GEFSv12_reforecast
source_dir=/glade/derecho/scratch/klesinger/GEFSv12_raw/${var}/${var}_processed
len_leads=35

python3 merge_GEFS_Northern_Hemisphere.py "${var}" "${save_dir}" "${source_dir}" "${len_leads}"

}

# merge_files_together "hgt_pres"


