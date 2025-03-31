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




## **📝 Usage**
### **🔹 Converting a DOCX file to MDX**
Run the script with:
```bash
python dump.py /path/to/input.docx rgb_or_hex_string
```
Example:
```bash
python dump.py "template/test_LIS.docx" "rgb"
```
or
```bash
python dump.py "template/test_LIS.docx" "hex"
```

This **automatically:**
- Extracts DOCX table and prose information
- Converts it into a **structured MDX file**
- Saves into `markdown/` directory

---


### **🔹 1. Color Conversion (Hex ↔ RGB)**
Automatically converts **colors between Hex and RGB** based on user preference.

🔹 **Function:** `color_converter()`
```python
def color_converter(color, hex_or_rgb="rgb"):
    """
    Converts Hex ↔ RGB based on user preference.
    """
```
✅ Converts `#FF5733` → `(255, 87, 51)`  
✅ Converts `rgb(255, 87, 51)` → `#FF5733`  
✅ **Keeps format intact** if already correct  

---

### **🔹 2. Converting DOCX file to MDX**
Extracts **table data, metadata, and prose blocks** while preserving formatting.

🔹 **Function:** `convert_docx_to_mdx_path()`
```python
def convert_docx_to_mdx_path(docx_path):
    """
    Converts a .docx file path to .data.mdx in 'converted_markdown'.
    """
```
- Creates `markdown/` folder
- **Renames `.docx` → `.data.mdx`**
- Saves the formatted MDX file

---


### **🔹 3. Adding Prose Blocks**
Dynamically appends prose sections without **overwriting existing content**.

🔹 **Function:** `add_prose_to_final_mdx()`
```python
def add_prose_to_final_mdx(outfile, prose_blocks):
    """
    Appends prose blocks while preserving spacing.
    """
```
✅ **Adds new `<Block>` sections**  
✅ **Maintains proper indentation**  
✅ **Prevents formatting corruption**

---

## **📂 Output Example**
Your **final MDX file** will look like this:

```mdx
---
id: lis-alaska-nrt
name: Land Information System - Alaska
description: State of Alaska vegetation and hydrological information produced by NASA’s
  Short-term Prediction and Transition Center – Land Information System (SPoRT-LIS).

layers:
  - id: alaska_relative_soil_moisture_10cm
    stacCol: lis_ak_rsm_10cm
    stacApiEndpoint: https://dev.openveda.cloud/api/stac
    name: Relative Soil Moisture (0-10cm), Updated Daily
    type: raster
    description: Relative soil moisture (RSM) is a ratio of the volumetric soil moisture
      between the wilting and saturation points for a given soil type.
    legend:
      unit:
        label: Percentage %
      type: gradient
      min: 0
      max: 100
      stops:
        - rgb(60,40,180)
        - rgb(111,96,219)
        - rgb(160,139,255)
        - rgb(149,209,251)
---

<Block>
  <Prose>
    **Temporal Extent:** 6 days prior - Present<br />
    **Temporal Resolution:** Daily<br />
    **Spatial Extent:** Alaska<br />
    **Spatial Resolution:** 0.03° x 0.03°<br />
    **Data Type:** Research<br />
    **Data Latency:** Updated Daily
  </Prose>
</Block>

<Block>
  <Prose>
    ## Source Data Product Citation
    Kumar, S.V., C.D. Peters-Lidard, Y. Tian, P.R. Houser, J. Geiger, S. Olden, L. Lighty, J.L. Eastman, B. Doty, P. Dirmeyer, J. Adams, K. Mitchell, E. F. Wood, and J. Sheffield.
  </Prose>
</Block>
```

---

## **📜 License**
This project is **open-source** under the **MIT License**.
