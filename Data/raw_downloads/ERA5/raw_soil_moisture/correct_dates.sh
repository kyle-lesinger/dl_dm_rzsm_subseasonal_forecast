#!/bin/bash

BASE_PATH="/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"

for YEAR in {1999..2021}; do
    for MONTH in {01..12}; do

        # Determine the last day of the month
        case $MONTH in
            01|03|05|07|08|10|12) LAST_DAY=31 ;;
            04|06|09|11) LAST_DAY=30 ;;
            02)
                # Check for leap year
                if (( (YEAR % 4 == 0 && YEAR % 100 != 0) || (YEAR % 400 == 0) )); then
                    LAST_DAY=29
                else
                    LAST_DAY=28
                fi
                ;;
        esac

        FILE="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_039_swvl1.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}${LAST_DAY}23.nc"
        FILE2="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_040_swvl2.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}${LAST_DAY}23.nc"
        FILE3="${BASE_PATH}/${YEAR}${MONTH}/e5.oper.an.sfc.128_041_swvl3.ll025sc.${YEAR}${MONTH}0100_${YEAR}${MONTH}${LAST_DAY}23.nc"
        echo "Processing: $FILE"

        # Example: check if file exists
        if [[ -f "$FILE" ]]; then
            echo "File exists: $FILE"
            cp $FILE .
            cp $FILE2 .
            cp $FILE3 .
            # Optional: cp "$FILE" /your/destination/
        else
            echo "File does not exist: $FILE"
        fi

    done
done
