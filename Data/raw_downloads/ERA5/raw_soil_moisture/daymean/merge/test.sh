#!/bin/bash

module load cdo

out_dir="scaled"
mkdir -p "$out_dir"

# Define input and output filenames
swvl3="swvl3.nc"

scaled3="$out_dir/swvl3_scaled.nc"
final_output="$out_dir/swvl_total.nc"

# Scale swvl3
if [[ -f "$scaled3" ]]; then
    echo "Skipping swvl3 scaling — $scaled3 already exists."
else
    echo "Scaling swvl3 by 0.72..."
    cdo mulc,0.72 "$swvl3" "$scaled3"
fi

