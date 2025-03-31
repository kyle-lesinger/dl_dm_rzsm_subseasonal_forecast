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



### **Features:**
- ✅ **Data Analysis** Can accomodate any reanalysis/reforecast as input if the data is an xarray object and you have a gridded mask file with the data coordinates that you want. (See /masks for example .nc4 and .grd files). Current datasets which have been studied within the manuscript include reanalysis products [GLEAM](https://www.gleam.eu/) and [ERA5-Land](https://www.ecmwf.int/en/era5-land); and subseasonal reforecast products [GEFSv12](https://vlab.noaa.gov/web/osti-modeling/gefsv12) and [ECMWF](https://apps.ecmwf.int/datasets/data/s2s/levtype=sfc/type=cf/).
- ✅ **Regional Training** Currently training was only completed on the contiguous United States, China, and Australia. But training can be altered if you have additional mask files (see /masks).



## **📜 License**
This project is **open-source** under the **MIT License**.
