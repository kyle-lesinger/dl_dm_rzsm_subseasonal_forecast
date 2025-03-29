#!/usr/bin/env python3
import xarray as xr
import functions as f
import numpy as np
import climpred
from xclim import sdba
import pandas as pd
import joypy

def setup_ridgeplot_array(metric_dict: dict, week_lead: int, region_name: str, metric_name: str, xlim_start: float, xlim_end:float) -> str:
    #Need to make the data into an array where:
    # ROWS = number of different realizations (so approximatley 15)
    # COLS = values (ACC or CRPS)

    save_dir = f'Outputs/joyplots/{region_name}'
    os.system(f'mkdir -p {save_dir}')

    df = pd.DataFrame(metric_dict).T

    if metric_name == 'CRPS':
        joypy.joyplot(df.T,colormap=cm.autumn_r,
                     title=f"{metric_name} by Experiment Week {week_lead}",
                     x_range=(0,0.05))
    else:
        joypy.joyplot(df.T,colormap=cm.autumn_r,
             title=f"{metric_name} by Experiment Week {week_lead}",
                     x_range=(xlim_start,xlim_end))
        
    plt.savefig(f'{save_dir}/Wk{week_lead}_{metric_name}.png')

    return('Completed')



