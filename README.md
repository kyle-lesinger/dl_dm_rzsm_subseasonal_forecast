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





## **📜 Authors** Kyle Lesinger, Di Tian

## **📜 License**
This project is **open-source** under the **MIT License**.
