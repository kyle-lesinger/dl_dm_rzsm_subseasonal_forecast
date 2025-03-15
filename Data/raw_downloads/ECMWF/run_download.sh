#!/bin/bash

module load conda
conda info --envs
conda activate /glade/work/klesinger/conda-envs/tf212gpu 



python download_data_update.py
