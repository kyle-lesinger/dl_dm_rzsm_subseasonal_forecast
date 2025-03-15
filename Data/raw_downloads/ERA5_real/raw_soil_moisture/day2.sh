#!/bin/bash

out="daymean"
mkdir -p "$out"

module load cdo

for file in *swvl*.nc; do
    outfile="$out/$file"
    if [[ -f "$outfile" ]]; then
        echo "Skipping $file — already processed."
    else
        echo "Processing $file → $outfile"
        cdo daymean "$file" "$outfile"
    fi
done
