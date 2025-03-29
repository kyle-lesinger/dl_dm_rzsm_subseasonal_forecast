#!/user/bin/env python3


def return_experiment_dictionaries(region_name):
    #####################These are models which only include the reforecast predictions. We are not including any observations into them.
    EX0 ={'region_name':region_name, 'num_lags_obs_RZSM': 0, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX13={'region_name':region_name, 'num_lags_obs_RZSM': 0, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':1 }
    
    
    ####################Include all predictors for obs but DO NOT include the current reforecast RZSM 
    EX1 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':0 }
    EX2 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':0 }
    EX3 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':0 }
    EX4 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':0 }
    
    
    #####################Include all predictors for obs but DO include the current reforecast RZSM 
    EX14 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': True, 'experiment_test':1 }
    EX15 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': True, 'experiment_test':2 }
    EX16 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':3 }
    EX17 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': False, 'addtl_experiment': False, 'experiment_test':4 }


    #####################Include only OBS RZSM predictors and the current reforecast RZSM week 
    EX5 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX6 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX7 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX8 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    
    ######################Include only OBS RZSM predictors, previous weeks prediction, and current reforecast RZSM week 
    EX18 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':1 }
    EX19 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':2 }
    EX20 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':3 }
    EX21 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':4 }
    
    ######################Include only OBS RZSM predictors DO NOT INCLUDE ANY REFORECAST OR PREDICTION
    EX22 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':5 }
    EX23 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':6 }
    EX24 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':7 }
    EX25 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': False, 'include_reforecast_or_not': True, 'addtl_experiment': True, 'experiment_test':8 }

    #######################Include only OBS RZSM predictors, OBS other variables, previous predictions of RZSM, and current reforecast lead
    EX9 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX10 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX11 ={'region_name':region_name, 'num_lags_obs_RZSM': 9, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }
    EX12 ={'region_name':region_name, 'num_lags_obs_RZSM': 12, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }


    ############################
    '''Ignore this experiment'''
    # EX26 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':0 }

    #######################This like EX9 and EX10 except now we are adding additional reforecast predictors from week 1 and 2
    EX27 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':1 }
    EX28 ={'region_name':region_name, 'num_lags_obs_RZSM': 6, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':1 }
    
    EX29 ={'region_name':region_name, 'num_lags_obs_RZSM': 3, 'include_lags_obs_pwat_spfh_tmax': True, 'include_reforecast_or_not': True, 'addtl_experiment': False, 'experiment_test':2 }

    return EX0,EX1,EX2,EX3,EX4,EX5,EX6,EX7,EX8,EX9,EX10,EX11,EX12,EX13,EX14,EX15,EX16,EX17,EX18,EX19,EX20,EX21,EX22,EX23,EX24,EX25,EX27,EX28,EX29

