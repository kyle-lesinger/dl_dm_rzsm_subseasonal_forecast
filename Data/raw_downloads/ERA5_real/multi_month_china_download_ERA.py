# %%
import cdsapi
import os
import xarray as xr
from multiprocessing import Pool 

# save_dir = os.getcwd()
save_dir = 'RAW_ERA5_china'
os.system(f'mkdir {save_dir}')

start_year = 1999
end_year = 2021 #(need to add 1 more year due to the range function)
year_list = [i for i in range(start_year,end_year)]


## Not a valid choice for ERA5 total_column_water_vapor, despite it being a 
#variable they say we can download online
var_list = ['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',]

#var_list=['maximum_2m_temperature_since_previous_post_processing']
#Make the month day list
day_list = []

for month in range(1,13):
    month = f'{month:02}'
    for year in range(1999,2021):
        day_list.append(f'{month}-{year}')

# %%
#Set up for multiprocessing
def download_data_by_day_month_year(month_year):
    for var in var_list:
        save_dir_new = f'{save_dir}/{var}'
        os.system(f'mkdir -p {save_dir_new}')
        
        month = month_year.split('-')[0]
        year = month_year.split('-')[1]
        
        out_name_of_file = f'{save_dir_new}/{var}_{month}-{year}.nc4'

        try:
            xr.open_dataset(out_name_of_file)
        except FileNotFoundError:
            print(f'Downloading {out_name_of_file}')
            c = cdsapi.Client()

            c.retrieve(
            'reanalysis-era5-land',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                f'{var}',
            ],
            'year': [
                f'{year}',
            ],
            'month': [
                f'{month}',
            ],
            'day': [
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
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                50, 30, 0,
                140,
            ],
        },
        f'{out_name_of_file}')

                
    return(0)



if __name__ == '__main__':
    p = Pool(1)
    p.map(download_data_by_day_month_year, day_list)

# %%
#Set up for multiprocessing
def download_data_by_day_month_year_geopotential(day_list):
    var = 'geopotential'
    save_dir_new = f'{save_dir}/{var}'
    os.system(f'mkdir -p {save_dir_new}')
    
    month = day_list.split('-')[0]
    year = day_list.split('-')[1]
    
    out_name_of_file = f'{save_dir_new}/{var}_{month}-{year}.nc4'

    try:
        xr.open_dataset(out_name_of_file)
    except FileNotFoundError:
        print(f'Downloading {out_name_of_file}')
        c = cdsapi.Client()

        c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'pressure_level': '200',
            'variable': f'{var}',
            
        'year': [
            f'{year}',
        ],
        'month': [
            f'{month}',
        ],
        'day': [
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
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            50, 30, 0,
            140,
        ],
    },
    f'{out_name_of_file}')

                
    return(0)



if __name__ == '__main__':
    p = Pool(1)
    p.map(download_data_by_day_month_year_geopotential, day_list)
