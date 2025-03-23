#!/bin/bash

module load conda
conda info --envs
conda activate /glade/work/klesinger/conda-envs/tf212gpu
module load ncl
module load cdo

# python soil_test.py
python multi_month_china_download_soil_ERA.py
