#!/bin/bash

module load cdo

out="merge"
mkdir -p "$out"

# Merge swvl1
if [[ -f "$out/swvl1.nc" ]]; then
    echo "Skipping swvl1 — $out/swvl1.nc already exists."
else
    echo "Merging swvl1 files..."
    cdo mergetime *swvl1* "$out/swvl1.nc"
fi

# Merge swvl2
if [[ -f "$out/swvl2.nc" ]]; then
    echo "Skipping swvl2 — $out/swvl2.nc already exists."
else
    echo "Merging swvl2 files..."
    cdo mergetime *swvl2* "$out/swvl2.nc"
fi

# Merge swvl3
if [[ -f "$out/swvl3.nc" ]]; then
    echo "Skipping swvl3 — $out/swvl3.nc already exists."
else
    echo "Merging swvl3 files..."
    cdo mergetime *swvl3* "$out/swvl3.nc"
fi
