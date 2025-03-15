# %%
import cdsapi
import os
import xarray as xr
from multiprocessing import Pool 

# save_dir = os.getcwd()
save_dir = 'RAW_ERA5_china'
os.system(f'mkdir {save_dir}')

start_year = 1999
end_year = 2022 #(need to add 1 more year due to the range function)
year_list = [i for i in range(start_year,end_year)]

#var_list=['maximum_2m_temperature_since_previous_post_processing']
#Make the month day list
day_list = []

for month in range(1,13):
    month = f'{month:02}'
    for year in range(1999,2021):
        day_list.append(f'{month}-{year}')


#Set up for multiprocessing
def download_data_by_day_month_year(month_year):

    var = 'soil_1m'
    save_dir_new = f'{save_dir}/{var}'
    os.system(f'mkdir -p {save_dir_new}')
    
    month = month_year.split('-')[0]
    year = month_year.split('-')[1]
    
    out_name_of_file = f'{save_dir_new}/{var}_{month}-{year}.nc4'

    try:
        xr.open_dataset(out_name_of_file)
    except FileNotFoundError:
        print(f'Downloading {out_name_of_file}')

        dataset = "reanalysis-era5-land"
        request = {
            "variable": [
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3"
            ],
            "year": f'{year}',
            "month": f'{month}',
            "day": [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
            ],
            "time": [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [45, 70, 18, 130]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()


                
    return(0)



if __name__ == '__main__':
    p = Pool(5)
    p.map(download_data_by_day_month_year, day_list)

