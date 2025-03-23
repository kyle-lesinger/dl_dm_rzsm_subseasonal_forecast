#!/bin/bash

#!/bin/bash

BASE_PATH="/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"

for YEAR in {1999..2021}; do
    for MONTH in {01..12}; do
        FILE="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_039_swvl1.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}3123.nc"
        echo "Processing: $FILE"
        
        # Example: check if file exists
        if [[ -f "$FILE" ]]; then
            echo "File exists: $FILE"
            # Optional action, e.g., cp "$FILE" /your/destination/
        else
            echo "File does not exist: $FILE"
        fi
    done
done

for YEAR in {1999..2021}; do
    for MONTH in {01..12}; do
        FILE="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_040_swvl2.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}3123.nc"
        echo "Processing: $FILE"
        
        # Example: check if file exists
        if [[ -f "$FILE" ]]; then
            echo "File exists: $FILE"
            # Optional action, e.g., cp "$FILE" /your/destination/
        else
            echo "File does not exist: $FILE"
        fi
    done
done

for YEAR in {1999..2021}; do
    for MONTH in {01..12}; do
        FILE="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_041_swvl3.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}3123.nc"
        echo "Processing: $FILE"
        
        # Example: check if file exists
        if [[ -f "$FILE" ]]; then
            echo "File exists: $FILE"
            # Optional action, e.g., cp "$FILE" /your/destination/
        else
            echo "File does not exist: $FILE"
        fi
    done
done