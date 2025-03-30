#!/bin/bash
#SBATCH --job-name=emos28             # job name
#SBATCH --nodes=1                    #number of nodes
#SBATCH --ntasks=14
#SBATCH --partition=general          # name of partition to submit job
#SBATCH --time=96:00:00              # Run time (D-HH:MM:SS)
#SBATCH --output=emos28.out          # Output file. %j is replaced with job ID
#SBATCH --error=emos28.err           # Error file. %j is replaced with job ID
#SBATCH --mail-type=ALL              # will send email for begin,end,fail
#SBATCH --mail-user=kdl0013@auburn.edu
#SBATCH --mem=40gb


#davisj8_std
#sleep 3600

#required modules
module load R/4.2.2

#Rscript emos_test_by_grid_cell_train_on_all_training_12_weeks_from_all_years_mclapply_easley28.R
#Rscript emos_test_by_grid_cell_train_on_all_training_12_weeks_before_and_after_from_all_years_mclapply_easley28.R
#Rscript emos_test_by_grid_cell_train_on_all_training_and_validation_mclapply_easley28.R
Rscript emos_test_by_grid_cell_train_on_all_training_mclapply_easley28.R

