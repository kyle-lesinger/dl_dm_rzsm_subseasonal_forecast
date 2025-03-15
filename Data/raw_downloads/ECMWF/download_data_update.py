# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 07:29:38 2024

@author: KDL0013
"""

from ecmwfapi import ECMWFDataServer
from glob import glob
from ecmwf.opendata import Client  #https://github.com/ecmwf/ecmwf-opendata/tree/main
import os
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
import pickle

use_multiprocessing = True
num_processes = 5


var_name = 'soilw_bgrnd'

os.system(f'mkdir -p {var_name}')


#To not flood the ECMWF portal with failed requests, I have saved the good model files into a text file

txt_name = 'good_model_dates.pickle'


#Create the hindcast date list

# Define start and end dates (need data 20 years back and these are the most recent model runs)
start_date = datetime(2015, 5, 14)
end_date = datetime(2024, 2, 22)

# Calculate the number of days between start and end dates
delta = end_date - start_date

# Create a list of all dates in the range
hindcast_date_list = [pd.to_datetime(start_date + timedelta(days=i)) for i in range(delta.days + 1)]

#Reverse the date list so that we get the most recent ones first
hindcast_date_list.reverse()

#Testing _date = hindcast_date_list[0]

#Now only grab the dates that are on Monday or Thursday (according to website documentation)

hindcast_date_list = [i for i in hindcast_date_list if i.dayofweek in [0, 3]]


# Step 1: Define the APIException class
class APIException(Exception):
    """Custom exception class for API errors."""
    def __init__(self, message="An error occurred with the API"):
        self.message = message
        super().__init__(self.message)
        
        
def return_hdates(_date):
    year_ = _date.year
    
    if (_date.month == 2) and (_date.day == 29):
        '''Must move back one day because of the leap year'''
        hdates = [f'{year_-i}-02-28' for i in range(1,21)]
    else:
        hdates = [f'{year_-i}-{_date.month:02}-{_date.day:02}' for i in range(1,21)]

    return(hdates)


#%%
#Now we have different leads seperated into 15 chunk slices
def download_by_day_control_RZSM(_date):
    print(f'\nTrying FIRST Date {_date}')
    #test 
    # _date = hindcast_date_list[0]
    
    dates_with_no_data = []
    good_model_dates = []
    
    var_name = 'soilw_bgrnd'
    
    realization = 'control'

    save_dir = f'/glade/derecho/scratch/klesinger/ECMWF/{var_name}'
    os.system(f'mkdir -p {save_dir}')
    
    
    #Now get the previous 20 years of the date
    #to not have to worry about the leap year, don't use time delta

    hdates = return_hdates(_date)

    for hdate in hdates:
        # break
        save_name = f'{save_dir}/{var_name}_{hdate}_{realization}.nc'
        
        if os.path.exists(save_name):
            pass
        else:
            if hdate[5:] in dates_with_no_data:
                break
            else:
                print(f'\nTrying date {hdate}')
                try:
                    server = ECMWFDataServer()
                    
                    server.retrieve({
                        "class": "s2",
                        "dataset": "s2s",
                        "date": f"{_date.year}-{_date.month:02}-{_date.day:02}",
                        "expver": "prod",
                        "hdate": hdate,
                        "levtype": "sfc",
                        "model": "glob",
                        "origin": "ecmf",
                        "param": "228087",
                        "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
                        "stream": "enfh",
                        "time": "00:00:00",
                        "type": "cf",
                        "target": save_name
                    })
                    good_model_dates.append(_date)
                    
                except APIException as e:
                    print('Cannot download this file')
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                except Exception as e:
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                    pass
                #This means that there is no data available for that date

    if len(good_model_dates) ==0:
        return()
    else:
        return(good_model_dates[0])
    
    
    
if __name__ == '__main__':
    if use_multiprocessing == True:
        p=Pool(num_processes)
        p.map(download_by_day_control_RZSM, hindcast_date_list)
        
        
#%%

#Now we have different leads seperated into 15 chunk slices
def download_by_day_control_other_variables(_date):

    #test 
    # _date = hindcast_date_list[0]
    
    dates_with_no_data = []
    good_model_dates = []
    
    var_name = 'temp_pwat_dewpoint'
    
    realization = 'control'

    save_dir = f'/glade/derecho/scratch/klesinger/ECMWF/{var_name}'
    os.system(f'mkdir -p {save_dir}')
    
    
    #Now get the previous 20 years of the date
    #to not have to worry about the leap year, don't use time delta

    year_ = _date.year
    
    hdates = return_hdates(_date)

    
    for hdate in hdates:
        # break
        save_name = f'{save_dir}/{var_name}_{hdate}_{realization}.nc'
        
        if os.path.exists(save_name):
            good_model_dates.append(_date)
            pass
        else:
            if hdate[5:] in dates_with_no_data:
                break
            else:
                print(f'\nTrying date {hdate}')
                try:
                    server = ECMWFDataServer()
                    
                    server.retrieve({
                        "class": "s2",
                        "dataset": "s2s",
                        "date": f"{_date.year}-{_date.month:02}-{_date.day:02}",
                        "expver": "prod",
                        "hdate": hdate,
                        "levtype": "sfc",
                        "model": "glob",
                        "origin": "ecmf",
                        "param": "136/167/168",
                        "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
                        "stream": "enfh",
                        "time": "00:00:00",
                        "type": "cf",
                        "target": save_name
                    })
                    good_model_dates.append(_date)
                    
                except APIException as e:
                    print('Cannot download this file')
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                except Exception as e:
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                    pass
                #This means that there is no data available for that date

    if len(good_model_dates) ==0:
        return()
    else:
        return(good_model_dates[0])



if __name__ == '__main__':
    if use_multiprocessing == True:
        p=Pool(num_processes)
        p.map(download_by_day_control_other_variables, hindcast_date_list)
        
        
        
#%%

#Now we have different leads seperated into 15 chunk slices
def download_by_day_perturbed_realization_RZSM(_date):
    var_name = 'soilw_bgrnd'

    save_dir = f'/glade/derecho/scratch/klesinger/ECMWF/{var_name}'
    os.system(f'mkdir -p {save_dir}')
    
    dates_with_no_data = []
    
    #Now get the previous 20 years of the date
    #to not have to worry about the leap year, don't use time delta

    year_ = _date.year
    
    hdates = return_hdates(_date)

    
    for hdate in hdates:
        # break
        save_name = f'{save_dir}/{var_name}_{hdate}_perturbed.nc'
        
        if os.path.exists(save_name):
            print('file exists')
            pass
        else:
            if hdate[5:] in dates_with_no_data:
                break
            else:
                print(f'\nTrying date {hdate}')
                try:
                    server = ECMWFDataServer()
                    
                    server.retrieve({
                        "class": "s2",
                        "dataset": "s2s",
                        "date": f"{_date.year}-{_date.month:02}-{_date.day:02}",
                        "expver": "prod",
                        "hdate": hdate,
                        "levtype": "sfc",
                        "model": "glob",
                        "origin": "ecmf",
                        "param": "228087",
                        "number": "1/2/3/4/5/6/7/8/9/10",
                        "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
                        "stream": "enfh",
                        "time": "00:00:00",
                        "type": "pf",
                        "target": save_name
                    })
                except APIException as e:
                    print('Cannot download this file')
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                except Exception as e:
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                    pass
                    #This means that there is no data available for that date
                    
        
        
#%%
#Now we have different leads seperated into 15 chunk slices
def download_by_day_perturbed_realization_other_variables(_date):
    
    var_name = 'temp_pwat_dewpoint'

    save_dir = f'/glade/derecho/scratch/klesinger/ECMWF/{var_name}'
    os.system(f'mkdir -p {save_dir}')
    
    dates_with_no_data = []
    
    #Now get the previous 20 years of the date
    #to not have to worry about the leap year, don't use time delta

    year_ = _date.year
    
    hdates = return_hdates(_date)

    for hdate in hdates:
        # break
        save_name = f'{save_dir}/{var_name}_{hdate}_perturbed.nc'
        
        if os.path.exists(save_name):
            # print('file exists')
            pass
        else:
            if hdate[5:] in dates_with_no_data:
                break
            else:
                # print(f'\nTrying date {hdate}')
                try:
                    server = ECMWFDataServer()
                    
                    server.retrieve({
                        "class": "s2",
                        "dataset": "s2s",
                        "date": f"{_date.year}-{_date.month:02}-{_date.day:02}",
                        "expver": "prod",
                        "hdate": hdate,
                        "levtype": "sfc",
                        "model": "glob",
                        "origin": "ecmf",
                        "param": "136/167/168",
                        "number": "1/2/3/4/5/6/7/8/9/10",
                        "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
                        "stream": "enfh",
                        "time": "00:00:00",
                        "type": "pf",
                        "target": save_name
                    })
                except APIException as e:
                    print('Cannot download this file')
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                except Exception as e:
                    dates_with_no_data.append(hdate[5:]) #Reduce the number of failed requests
                    print(f'Not downloading anymore dates for {hdate}')
                    pass
                    #This means that there is no data available for that date
                    
                    

if __name__ == '__main__':
    if use_multiprocessing == True:
        p=Pool(num_processes)
        p.map(download_by_day_perturbed_realization_RZSM, hindcast_date_list)
        p.map(download_by_day_perturbed_realization_other_variables, hindcast_date_list)
        


    
    
