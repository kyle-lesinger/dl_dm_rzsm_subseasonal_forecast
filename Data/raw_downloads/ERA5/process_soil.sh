#!/bin/bash

module load cdo

# Loop through all NetCDF (.nc) files in the current directory
for file in *.nc; do
    # Define output filename
    output_file="final_${file%.nc}.nc4"

    # Check if the final output already exists
    if [ -f "$output_file" ]; then
        echo "✅ $output_file already exists. Skipping..."
        continue
    fi

    echo "🔄 Processing $file..."

    # Step 1: Calculate daily mean
    daymean_file="daymean_${file%.nc}.nc"
    cdo daymean "$file" "$daymean_file"

    # Step 2: Scale each variable
    swvl1_scaled="swvl1_scaled_${file%.nc}.nc"
    swvl2_scaled="swvl2_scaled_${file%.nc}.nc"
    swvl3_scaled="swvl3_scaled_${file%.nc}.nc"

    cdo mulc,0.07 -selname,swvl1 "$daymean_file" "$swvl1_scaled"
    cdo mulc,0.21 -selname,swvl2 "$daymean_file" "$swvl2_scaled"
    cdo mulc,0.72 -selname,swvl3 "$daymean_file" "$swvl3_scaled"

    # Step 3: Add the scaled variables
    temp_add="temp_add_${file%.nc}.nc"
    cdo add "$swvl1_scaled" "$swvl2_scaled" "$temp_add"
    sum_file="sum_${file%.nc}.nc"
    cdo add "$temp_add" "$swvl3_scaled" "$sum_file"

    # Step 4: Rename variable and save final output as .nc4
    cdo chname,swvl1,scaled_soil_moisture "$sum_file" "$output_file"

    # Step 5: Clean up intermediate files
    rm "$daymean_file" "$swvl1_scaled" "$swvl2_scaled" "$swvl3_scaled" "$temp_add" "$sum_file"

    echo "✅ Done: $output_file"
done

