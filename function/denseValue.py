import numpy as np

def return_dense_value_file(region_name,density_value):
    # density = np.load('Data/model_npy_inputs/weighted_density/weighted_density_0.8.npy')
    density = np.load(f'Data/model_npy_inputs/weighted_density/weighted_density_{region_name}_{density_value}.npy')
    np.save('Data/density.npy',density)
    return(0)


