#!/bin/bash

module load cdo

out_dir="scaled"
mkdir -p "$out_dir"

# Define input and output filenames
swvl1="swvl1.nc"
swvl2="swvl2.nc"
swvl3="swvl3.nc"

scaled1="$out_dir/swvl1_scaled.nc"
scaled2="$out_dir/swvl2_scaled.nc"
scaled3="$out_dir/swvl3_scaled.nc"
final_output="$out_dir/swvl_total.nc"

# Scale swvl1
if [[ -f "$scaled1" ]]; then
    echo "Skipping swvl1 scaling — $scaled1 already exists."
else
    echo "Scaling swvl1 by 0.07..."
    cdo mulc,0.07 "$swvl1" "$scaled1"
fi

# Scale swvl2
if [[ -f "$scaled2" ]]; then
    echo "Skipping swvl2 scaling — $scaled2 already exists."
else
    echo "Scaling swvl2 by 0.21..."
    cdo mulc,0.21 "$swvl2" "$scaled2"
fi

# Scale swvl3
if [[ -f "$scaled3" ]]; then
    echo "Skipping swvl3 scaling — $scaled3 already exists."
else
    echo "Scaling swvl3 by 0.72..."
    cdo mulc,0.72 "$swvl3" "$scaled3"
fi

# Add all scaled layers
if [[ -f "$final_output" ]]; then
    echo "Skipping final addition — $final_output already exists."
else
    echo "Adding scaled swvl1 + swvl2 + swvl3 → $final_output"
    cdo add "$scaled1" "$scaled2" "$out_dir/temp_add.nc"
    cdo add "$out_dir/temp_add.nc" "$scaled3" "$final_output"
    rm "$out_dir/temp_add.nc"
fi
