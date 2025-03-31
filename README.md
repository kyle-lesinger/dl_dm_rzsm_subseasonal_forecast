# **Skillful Subseasonal Soil Moisture Drought Forecasts with Deep Learning-Dynamic Models**


## **📌 Overview**
This set of python script and Jupyter notebooks will allow for the processing and visualization of Earth data for the research article Skillful Subseasonal Soil Moisture Drought Forecasts with Deep Learning-Dynamic Models. These scripts are set to be run in order (beginning at script 00.ipynb ---> 09.ipynb).


## **📦 Installation**
### **🔹 Install in new conda environment**
```bash
conda env create -f conda_environment_setup.yaml
conda activate tf212gpu_new
```

## **📌 Restrictions**
1.) For training the deep-learning models, you must be access to a GPU with at least 32GB RAM.
2.) For some other functions, as high as 80GB RAM may be needed for pre-processing.



### ** 📌 Features:**
- ✅ **Data Analysis** Can accomodate any reanalysis/reforecast as input if the data is an xarray object and you have a gridded mask file with the data coordinates that you want. (See /masks for example .nc4 and .grd files). Current datasets which have been studied within the manuscript include reanalysis products [GLEAM](https://www.gleam.eu/), [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5), [ERA5-Land](https://www.ecmwf.int/en/era5-land); and subseasonal reforecast products [GEFSv12](https://vlab.noaa.gov/web/osti-modeling/gefsv12) and [ECMWF](https://apps.ecmwf.int/datasets/data/s2s/levtype=sfc/type=cf/).
- ✅ **Regional Training** Currently training was only completed on the contiguous United States, China, and Australia. But training can be altered if you have additional mask files (see /masks).
- ✅ **Setting path locations for data** Use the /function/conf.py to setup absolute paths. Can also add additional datapaths if new sources are added.


### ** 📌 Downloading Data:**
-  [GLEAM](https://www.gleam.eu/). Must contact developers and get the sftp information. Must save into directory 
-  [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5). Must use CDS (Climate Data Store) and create your own credentials. 
-  [ERA5-Land](https://www.ecmwf.int/en/era5-land).  Must use CDS (Climate Data Store) and create your own credentials.
-  [GEFSv12](https://vlab.noaa.gov/web/osti-modeling/gefsv12). Can download with /Data/raw_downloads/GEFSv12 scripts. Use the run_parallel_all_regions.sh as the run file.
-  [ECMWF](https://apps.ecmwf.int/datasets/data/s2s/levtype=sfc/type=cf/). Can download with /Data/raw_downloads/ECMWF scripts. Follow the order of the Jupyter notebooks.

### ** 📌 Description of scripts and their purpose:**
- 1.) 00_min_max_each_region_&reforecast.ipynb - Convert each data type to the same format across different data sources. Saves anomalies, and creates files formatted to work with tensorflow during training.
- 2.) 01_make_small_plots_for_diagram.ipynb - Plots some of the files to ensure that they look acceptable.
- 3.) 01a_bias_correct_raw_GEFS_ECMWF.ipynb - Bias correct raw GEFSv12 and ECMWF files (use additive mean bias correction).
- 4.) 02_run_model_EXPERIMENTS_ECMWF_and_GEFSv12_v5.ipynb - Train deep learning models. Save predictions. Compute permutation test.
- 5.) 03a_save_ACC_for_each_lead_and_experiment.ipynb - Save anomaly correlation coefficient values (ACC) for each experiment type.
- 6.) 03b_save_CRPSS_for_each_lead_and_experiment.ipynb - Save continous ranked probability score values (CRPS) for each experiment type.
- 7.) 04a_ACC_CRPSS_hitRate_CONUS.ipynb - Create plots for the ACC and CRPS for each experiment type. Spatial plots and line plots included. Only for CONUS region which included multiple experiments.
- 8.) 04b_ACC_CRPSS_hitRate_season_CONUS.ipynb - Create plots for the ACC and CRPS for each experiment type. Spatial plots and line plots included. Only for China and Australia.
- 9.) 04c_KGE_CONUS.ipynb - Save Kling-Gupte Efficiency (KGE) for CONUS.
- 10.) 04d_Plot_permutation_test_results_UPDATE.ipynb - Plot the CONUS permutation tests.
- 11.) 04e_autocorrelation_GLEAM.ipynb - Compute and plot the autocorrelation for each region.
- 12.) 05a_create_percentile_FULL_DISTRIBUTION_OBS_GEFS_ECMWF_v9_multi.ipynb - Computes percentiles of the data from the training distribution (2000-2015) and creates a percentile of score with the testing distribution (2018-2019).
- 13.) 05b_create_percentile_SMALL_DISTRIBUTION_OBS_GEFS_ECMWF_bias_corrected_v1.ipynb - Computes the percentile of score with only the testing distribution (2018-2019).
- 14.) 05c_hitRate_CONUS.ipynb - Computes the true positive rate (aka the hit rate) for data according to a specific percentile distribution that is user selected.
- 15.) 06a_ACC_spatial_plots_other_regions.ipynb - Plot spatial ACC results for a single experiment across different regions. Data **is not** separated by season.
- 16.) 06b_ACC_CRPS_season_other_regions.ipynb - Plot spatial ACC results for a single experiment across different regions. Data **is** separated by season.




## **📜 Authors** Kyle Lesinger, Di Tian

## **📜 License**
This project is **open-source** under the **MIT License**.
