#!/usr/bin/env python3

import os
import numpy as np
from function import addPredictors as pred
from function import funs as f

def load_all_data_EX0(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both,addtl_experiment,experiment_test):

    soil_var = 'soilw_bgrnd' #This is the observation verification variable name
    
    channel_list = []

    if addtl_experiment == False:
        assert lead >=1, 'Lead must be >=1, do not look at Week 0'
        
        lead_list = np.arange(1,lead+1)

        print(f'Only adding Reforecast data as input for leads {list(lead_list)}.')

        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],lead)) #We are adding both RZSM and/or Tmax (so use * 2)
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead))

        #Add reforecast data
        for idx,lead in enumerate(lead_list):
            channel_list.append(f'RZSM_ref_lead{lead}')

            training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx,ref_source,final_testing_year)

        print(f'Index idx value is {idx}. Done adding RZSM obs.')    

    
    elif addtl_experiment == True:
        print(f'Only adding Reforecast data as input for lead {lead}.')

        lead_add = 1 #We are only adding the current week of reforecast
        lead_list = [lead]

        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],lead_add)) #We are adding both RZSM and Tmax (so use * 2)
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead_add))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],lead_add))

        channel_list.append(f'RZSM_ref_lead{lead}')

        idx = 0
        training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx,ref_source,final_testing_year)

    return(training_input, validation_input, testing_input, channel_list)

def load_all_data_EX1_EX2_EX3_EX4_after_week_0(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, 
                                           observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,
                                           validation_testing_size_shape,experiment_name,RZSM_or_Tmax_or_both,addtl_experiment,
                                               experiment_test, experiment_name_out,ref_source,final_testing_year,obs_source):
    channel_list = []
    
    #Reforecast predictors
    var_list = ['pwat_eatm', 'spfh_2m', 'tmax_2m', 'diff_temp_2m', 'hgt_pres']

    assert lead >=1, 'Lead must be >=1, we are not testing week 0 anymore'

    if addtl_experiment == False:
        #Get the number of extra channels to add from previous weeks predictions
        if RZSM_or_Tmax_or_both == 'both':
            add_channels = lead * 2
        else:
            #Also add the number of channels according to the lead. If lead == 1, then only add lead 1 current week. If lead == 2, add one channel for week 2 and one channel for the prediction from Week 1
            if lead > 1:
                add_channels = lead - 1
            else:
                add_channels = 0

        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))

        print(f'Training shape input is {training_input.shape}')

        #Add RZSM observations first
        for idx,lag in enumerate(lag_integer_list):
            channel_list.append(f'RZSM_obs_lag{lag}')
            training_input, validation_input, testing_input = pred.add_obs_RZSM_by_lag(training_input, validation_input, testing_input, lag, input_directory, idx, final_testing_year, obs_source)

        print(f'Index idx value is {idx}. Done adding RZSM obs.')
        
        RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory, final_testing_year,obs_source)

        
        for variable in var_list:
            channel_name_output = pred.return_channel_name(variable)
            for lag in observation_lag_list_not_RZSM:
                channel_list.append(f'{channel_name_output}_obs_lag{lag}')
                idx+=1
                #Observations adding
                training_input, validation_input, testing_input = pred.add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx, final_testing_year)
     
        
            print(f'Index idx value is {idx}. Done adding {variable} obs.')
    
        if lead in [0,1]:
            pass
        else:
            #Now we need to load our data from the previous weeks
            for lead_previous in range(1,lead):
                print(f'Working on previous lead: Week {lead_previous}')
                # break
    
                pred_dir = f'predictions/{region_name}/Wk{lead_previous}_training_validation'
                testing_dir = f'predictions/{region_name}/Wk{lead_previous}_testing'
    
                os.system(f'mkdir -p {pred_dir}')
    
                training_prediction_name= f'{pred_dir}/Wk{lead_previous}_training_{experiment_name_out}.npy'
                validation_prediction_name = f'{pred_dir}/Wk{lead_previous}_validation_{experiment_name_out}.npy'
                testing_prediction_name = f'{testing_dir}/Wk{lead_previous}_testing_{experiment_name_out}.npy'
    
                testing_prediction = np.load(testing_prediction_name)
                
                '''Load the training and validation predictions if they exist'''
                if os.path.exists(training_prediction_name):
                    print(f'Loading data from previous week lead {lead_previous}')
                    training_prediction = np.load(training_prediction_name)
                    validation_prediction =np.load(validation_prediction_name)
    
                else:
                    print(f'Creating prediction data from previous week lead {lead_previous}')
                    '''If they don't exist..... load model for previous week'''
                    model = load_model(f'checkpoints/Wk{lead_previous}/Wk{lead_previous}_{experiment_name_out}',compile=False) #don't need the custom loss function for predictions
    
                    training_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_training_input.npy')
                    validation_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_validation_input.npy')
                    testing_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_testing_input.npy')
    
                    '''Make predictions and save data to load for previous use'''
                    training_prediction = model.predict(training_input_previous) #Just for unit checking
                    np.save(training_prediction_name,training_prediction)
    
                    validation_prediction = model.predict(validation_input_previous)
                    np.save(validation_prediction_name,validation_prediction)
    
                ############# ADD BACK TO CREATED ARRAY ###############################
                '''Now get the best UNET index prediction for RZSM and Tmax. Because the UNET plus plus produces 3-4 different outputs'''
                min_RZSM_index,min_tmax_index = f.load_loss_csv_bias_correction(lead_previous,experiment_name_out,RZSM_or_Tmax_or_both,region_name)
    
                if np.array(training_prediction).shape[0] == 8 and RZSM_or_Tmax_or_both == 'both':
                    min_tmax_index = min_tmax_index+4 #Because index values 0-3 are RZSM and 4-7 are Tmax
                elif np.array(training_prediction).shape[0] == 6 and RZSM_or_Tmax_or_both == 'both':
                    min_tmax_index = min_tmax_index + 3 #Because index values are 0-2 for RZSM and 3-5 for Tmax
                '''Now split the training and validation data back into RZSM and/or Tmax'''
    
    
                train_RZSM = np.array(training_prediction)[min_RZSM_index,:,:,:,:]
                val_RZSM = np.array(validation_prediction)[min_RZSM_index,:,:,:,:]
                test_RZSM = np.array(testing_prediction)[min_RZSM_index,:,:,:,:]

    
                '''Make sure we mask RZSM values which aren't on land'''
                train_RZSM = np.where(RZSM_train_obs==0,0,train_RZSM)
                val_RZSM = np.where(RZSM_validation_obs==0,0,val_RZSM)
                test_RZSM = np.where(RZSM_validation_obs==0,0,test_RZSM)
    
                #Now add back to the data
                '''Now add back to the newly created dataset'''
                idx+=1
                channel_list.append(f'RZSM_prediction_lead{lead_previous}')
                print(f'Adding RZSM training, validation, testing into index {idx}')
                training_input[:,:,:,idx] = train_RZSM[:,:,:,0]
                validation_input[:,:,:,idx] = val_RZSM[:,:,:,0]
                testing_input[:,:,:,idx] = test_RZSM[:,:,:,0]
    
            if RZSM_or_Tmax_or_both == 'both':
                channel_list.append(f'tmax_prediction_lead{lead_previous}')
                idx+=1
                print(f'Adding Tmax training, validation, testing into index {idx}')
                training_input[:,:,:,idx] = train_tmax[:,:,:,0]
                validation_input[:,:,:,idx] = val_tmax[:,:,:,0]
                testing_input[:,:,:,idx] = test_tmax[:,:,:,0]
    

            print('Final channel of input is the following (just make sure it has zeros for RZSM)')
            print(training_input[0,:,:,-1])

    elif addtl_experiment == True:
        #Get the number of extra channels to add from previous weeks predictions
        add_channels = 0


        training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+add_channels))
        validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+add_channels))
        testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(observation_lag_list_not_RZSM)*5+add_channels))


        #Add RZSM observations first
        for idx,lag in enumerate(lag_integer_list):
            channel_list.append(f'RZSM_obs_lag{lag}')
            training_input, validation_input, testing_input = pred.add_obs_RZSM_by_lag(training_input, validation_input, testing_input, lag, input_directory, idx, final_testing_year, obs_source)

        print(f'Index idx value is {idx}. Done adding RZSM obs.')    
        
        RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory, final_testing_year,obs_source)


        for variable in var_list:
            channel_name_output = pred.return_channel_name(variable)
            for lag in observation_lag_list_not_RZSM:
                channel_list.append(f'{channel_name_output}_obs_lag{lag}')
                idx+=1
                #Observations adding
                training_input, validation_input, testing_input = pred.add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx, final_testing_year)
     
        
            print(f'Index idx value is {idx}. Done adding {variable} obs.')

        print('Final channel of input is the following (just make sure it has zeros for RZSM)')
        print(training_input[0,:,:,-1])

    return(training_input, validation_input, testing_input,channel_list)



def load_all_data_EX5_EX6_EX7_EX8_experiments(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, 
                                              include_reforecast_or_not, observation_lag_list_not_RZSM, lag_integer_list,
                                              input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both,experiment_test,experiment_name,
                                              experiment_name_out,ref_source,final_testing_year,obs_source):
    print('\nUsing observations as predictions and forecast lead week 1\n')
    #Observations RZSM
    '''We need to combine all the RZSM files into 1 single dictionary for later processing'''
    
    channel_list = []

    assert lead >=1, 'We are only looking at weekly leads 1-5, cannot do 0 lead'
    
    #Reforecast predictors
    var_list = ['pwat_eatm', 'spfh_2m', 'tmax_2m', 'diff_temp_2m', 'hgt_pres']
    
    print(f'Experiment name is {experiment_name_out}')
    
    if experiment_name in ['EX22','EX23','EX24','EX25']:
        #This is only a bias correction using observations (no reforecast input at all)
        lead_add = 0
    elif experiment_name in ['EX5','EX6','EX7','EX8']:
        #This is only a bias correction using observations and the current week of reforecast as inputs
        lead_add = 1
    elif experiment_name in ['EX18','EX19','EX20','EX21']:
        if lead ==1:
            lead_add = 1 #Only adding the current week of forecast
        else:
            lead_add = lead #Adding previous weeks forecast and the current week

    print(f'Adding {lead_add} to channel list from reforecast.')

        
    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+ lead_add))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list) + lead_add))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+ lead_add))
    
    print(f'Training shape: {training_input.shape}')
    
    #Add RZSM observations first
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        training_input, validation_input, testing_input = pred.add_obs_RZSM_by_lag(training_input, validation_input, testing_input, lag, input_directory, idx, final_testing_year, obs_source)

    print(f'Index idx value is {idx}. Done adding RZSM obs.')
    
    #For masking
    RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory, final_testing_year,obs_source)

    if experiment_name in ['EX5','EX6','EX7','EX8']:
        #We only need to add the current reforecast week
        #Add final predictor for the current week of RZSM within reforecast
        idx+=1
        soil_var = 'soilw_bgrnd'
        channel_name_output = pred.return_channel_name(soil_var)
        channel_list.append(f'{channel_name_output}_ref_lead{lead}')
    
        print(f'And finally adding RZSM week {lead} from reforecast as a predictor')
        
        training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx, ref_source,final_testing_year)
        
        return(training_input, validation_input, testing_input, channel_list)

    elif experiment_name in ['EX22','EX23','EX24','EX25']:
         #We are not adding any additional predictors
         return(training_input, validation_input, testing_input, channel_list)
        
    
    else:
        assert lead >= 2, 'If trying to run EX18-EX21, DO NOT RUN with week 1 data. Start with Week 2'



        #Now we need to load our data from the previous weeks
        for lead_previous in range(1,lead):

            #To not have to re-run a bunch more models, we can re-use the training from a previous week
            if lead_previous == 1:
                if experiment_name == 'EX18':
                    prev_experiment = f'EX5_{save_loss_name}_RZSM'
                        
                elif experiment_name == 'EX19':
                    prev_experiment = f'EX6_{save_loss_name}_RZSM'    
                    
                elif experiment_name == 'EX20':
                    prev_experiment = f'EX7_{save_loss_name}_RZSM'  
                    
                elif experiment_name == 'EX21':
                    prev_experiment = f'EX8_{save_loss_name}_RZSM'
            elif lead > 2:
                prev_experiment = experiment_name_out
            
            print(f'Working on previous lead: Week {lead_previous}')
            # break

            pred_dir = f'predictions/{region_name}/Wk{lead_previous}_training_validation'
            testing_dir = f'predictions/{region_name}/Wk{lead_previous}_testing'

            os.system(f'mkdir -p {pred_dir}')

            training_prediction_name= f'{pred_dir}/Wk{lead_previous}_training_{prev_experiment}.npy'
            validation_prediction_name = f'{pred_dir}/Wk{lead_previous}_validation_{prev_experiment}.npy'
            testing_prediction_name = f'{testing_dir}/Wk{lead_previous}_testing_{prev_experiment}.npy'


            testing_prediction = np.load(testing_prediction_name)
            '''Load the training and validation predictions if they exist'''
            
            if os.path.exists(training_prediction_name):
                print(f'Loading data from previous week lead {lead_previous}')
                training_prediction = np.load(training_prediction_name)
                validation_prediction =np.load(validation_prediction_name)

            else:
                print(f'Creating prediction data from previous week lead {lead_previous} from {prev_experiment}.')
                '''If they don't exist..... load model for previous week'''
                model = load_model(f'checkpoints/Wk_{lead_previous}/Wk{lead_previous}_{prev_experiment}',compile=False) #don't need the custom loss function for predictions

                training_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{prev_experiment}_training_input.npy')
                validation_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{prev_experiment}_validation_input.npy')
                testing_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{prev_experiment}_testing_input.npy')

                '''Make predictions and save data to load for previous use'''
                training_prediction = model.predict(training_input_previous) #Just for unit checking
                np.save(training_prediction_name,training_prediction)

                validation_prediction = model.predict(validation_input_previous)
                np.save(validation_prediction_name,validation_prediction)

            ############# ADD BACK TO CREATED ARRAY ###############################
            '''Now get the best UNET index prediction for RZSM and Tmax. Because the UNET plus plus produces 4 different outputs'''
            min_RZSM_index,min_tmax_index = f.load_loss_csv_bias_correction(lead_previous,prev_experiment,RZSM_or_Tmax_or_both, region_name)

            if np.array(training_prediction).shape[0] == 8 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index+4 #Because index values 0-3 are RZSM and 4-7 are Tmax
            elif np.array(training_prediction).shape[0] == 6 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index + 3 #Because index values are 0-2 for RZSM and 3-5 for Tmax
            '''Now split the training and validation data back into RZSM and/or Tmax'''


            train_RZSM = np.array(training_prediction)[min_RZSM_index,:,:,:,:]
            val_RZSM = np.array(validation_prediction)[min_RZSM_index,:,:,:,:]
            test_RZSM = np.array(testing_prediction)[min_RZSM_index,:,:,:,:]


            '''Make sure we mask RZSM values which aren't on land'''
            train_RZSM = np.where(RZSM_train_obs==0,0,train_RZSM)
            val_RZSM = np.where(RZSM_validation_obs==0,0,val_RZSM)
            test_RZSM = np.where(RZSM_validation_obs==0,0,test_RZSM)

            #Now add back to the data
            '''Now add back to the newly created dataset'''
            idx+=1
            channel_list.append(f'RZSM_prediction_lead{lead_previous}')
            print(f'Adding RZSM training, validation, testing into index {idx}')
            training_input[:,:,:,idx] = train_RZSM[:,:,:,0]
            validation_input[:,:,:,idx] = val_RZSM[:,:,:,0]
            testing_input[:,:,:,idx] = test_RZSM[:,:,:,0]


        #Add final predictor for the current week of RZSM within reforecast
        idx+=1
        soil_var = 'soilw_bgrnd'
        channel_name_output = pred.return_channel_name(soil_var)
        channel_list.append(f'{channel_name_output}_ref_lead{lead}')
    
        print(f'And finally adding RZSM week {lead} from reforecast as a predictor')
        
        training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx, ref_source,final_testing_year)
    
        print(f'Index idx value is {idx}. Done adding {soil_var} reforecast.')

        print('Final channel of input is the following (just make sure it has zeros for RZSM)')
        print(training_input[0,:,:,-1])


        return(training_input, validation_input, testing_input, channel_list)



def load_all_data_EX9_EX10_EX11_EX12_after_week_0(lead, num_lags_obs_RZSM, include_lags_obs_pwat_spfh_tmax, include_reforecast_or_not, 
                                           observation_lag_list_not_RZSM, lag_integer_list, input_directory,training_size_shape,
                                           validation_testing_size_shape,experiment_name,RZSM_or_Tmax_or_both, experiment_test,
                                                  region_name,experiment_name_out, final_testing_year,obs_source,ref_source):
    '''We need to combine all the RZSM files and all the observation data (pwat, spfh, tmax, diff_temp, z200) into one file. Also adding the RZSM reforecast data from RZSM and Tmax '''
    
    
    channel_list = []
    
    #Reforecast predictors
    var_list = ['pwat_eatm', 'spfh_2m', 'tmax_2m', 'diff_temp_2m', 'hgt_pres']
    
    if ref_source == 'GEFSv12':
        var_list_ref = var_list
    elif ref_source == 'ECMWF':
        var_list_ref = ['t2m', 'd2m', 'tcw']

    if experiment_test == 0:
        add_channels = lead  
        reforecast_predictors = False
    elif (experiment_test >= 1) and (lead in [1,2]):
        #This is adding 5 additional predictors (tmax, diff_temp, z200, pwat, spfh)
        #Also add the number of channels according to the lead. If lead == 1, then only add lead 1 current week. If lead == 2, add one channel for week 2 and one channel for the prediction from Week 1
        add_channels = lead  + len(var_list_ref)
        reforecast_predictors = True
    elif (experiment_test >= 1) and (lead > 2):
        add_channels = lead 
        reforecast_predictors = False
            
   
    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],len(lag_integer_list)+len(var_list)*3+add_channels))
    
    print(f'Training input shape = {training_input.shape}')
    
    #Add RZSM observations first
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        training_input, validation_input, testing_input = pred.add_obs_RZSM_by_lag(training_input, validation_input, testing_input, lag, input_directory, idx,final_testing_year, obs_source)
        
    print(f'Index idx value is {idx}. Done adding RZSM obs.')    
    
    RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory,final_testing_year)

    for variable in var_list:
        channel_name_output = pred.return_channel_name(variable)
        for lag in observation_lag_list_not_RZSM:
            channel_list.append(f'{channel_name_output}_obs_lag{lag}')
            idx+=1
            #Observations adding
            training_input, validation_input, testing_input = pred.add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx,final_testing_year)
 
    
        print(f'Index idx value is {idx}. Done adding {variable} obs.')
    
    ############################### NOW ADD THE PREDICTION DATA FROM PREVIOUS WEEK ################################################
    #Now we need to load our data from the previous weeks 
    
    if lead in [0, 1]:
        #DO NOT INCLUDE LEAD 0 AS A PREDICTOR (FROM PREVIOUS WEEK)
        pass
    else:

        for lead_previous in range(1,lead):
            print(f'Adding previous RZSM prediction from week {lead_previous} as an input channel')
            # break
            
            pred_dir = f'predictions/{region_name}/Wk{lead_previous}_training_validation'
            testing_dir = f'predictions/{region_name}/Wk{lead_previous}_testing'
            
            os.system(f'mkdir -p {pred_dir}')
            
            training_prediction_name= f'{pred_dir}/Wk{lead_previous}_training_{experiment_name_out}.npy'
            validation_prediction_name = f'{pred_dir}/Wk{lead_previous}_validation_{experiment_name_out}.npy'
            testing_prediction_name = f'{testing_dir}/Wk{lead_previous}_testing_{experiment_name_out}.npy'
            
            testing_prediction = np.load(testing_prediction_name)
            
            '''Load the training and validation predictions if they exist'''
            if os.path.exists(training_prediction_name):
                print(f'Loading data from previous week lead {lead_previous}')
                training_prediction = np.load(training_prediction_name)
                validation_prediction =np.load(validation_prediction_name)
                
            else:
                print(f'Creating prediction data from previous week lead {lead_previous}')
                '''If they don't exist..... load model for previous week'''
                model = load_model(f'checkpoints/Wk_{lead_previous}/Wk{lead_previous}_{experiment_name_out}',compile=False) #don't need the custom loss function for predictions
                
                training_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_training_input.npy')
                validation_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_validation_input.npy')
                testing_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_testing_input.npy')
                
                '''Make predictions and save data to load for previous use'''
                training_prediction = model.predict(training_input_previous) #Just for unit checking
                np.save(training_prediction_name,training_prediction)
                
                validation_prediction = model.predict(validation_input_previous)
                np.save(validation_prediction_name,validation_prediction)
            
            
            ############# ADD BACK TO CREATED ARRAY ###############################
            '''Now get the best UNET index prediction for RZSM and Tmax. Because the UNET plus plus produces 4 different outputs'''
            min_RZSM_index,min_tmax_index = f.load_loss_csv_bias_correction(lead_previous,experiment_name_out,RZSM_or_Tmax_or_both,region_name)
    
            if np.array(training_prediction).shape[0] == 8 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index+4 #Because index values 0-3 are RZSM and 4-7 are Tmax
            elif np.array(training_prediction).shape[0] == 6 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index + 3 #Because index values are 0-2 for RZSM and 3-5 for Tmax
            '''Now split the training and validation data back into RZSM and/or Tmax'''
            
            
            train_RZSM = np.array(training_prediction)[min_RZSM_index,:,:,:,:]
            val_RZSM = np.array(validation_prediction)[min_RZSM_index,:,:,:,:]
            test_RZSM = np.array(testing_prediction)[min_RZSM_index,:,:,:,:]

    
            '''Make sure we mask RZSM values which aren't on land'''
            train_RZSM = np.where(RZSM_train_obs==0,0,train_RZSM)
            val_RZSM = np.where(RZSM_validation_obs==0,0,val_RZSM)
            test_RZSM = np.where(RZSM_validation_obs==0,0,test_RZSM)
    
            #Now add back to the data
            '''Now add back to the newly created dataset'''
            idx+=1
            channel_list.append(f'RZSM_prediction_lead{lead_previous}')
            print(f'Adding RZSM training, validation, testing into index {idx}')
            training_input[:,:,:,idx] = train_RZSM[:,:,:,0]
            validation_input[:,:,:,idx] = val_RZSM[:,:,:,0]
            testing_input[:,:,:,idx] = test_RZSM[:,:,:,0]

    
    ############################### NOW ADD THE REFORECAST DATA FROM THE WEEK TO BE PREDICTED ################################################
    
    #Add predictors from the reforecast model
    if (lead in [1,2]) and (experiment_test == 1):
        print(f'Adding current reforecast data from week {lead} for vars {(var_list_ref)}')  
        #Only add these for week 1 and 2. 
        for variable in var_list_ref:
            idx+=1
            channel_name_output = pred.return_channel_name(variable)
            
            channel_list.append(f'{channel_name_output}_ref_lead{lead}')
            
            training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory, variable, idx, ref_source,final_testing_year)
    
            print(f'Index idx value is {idx}. Done adding {variable} reforecast.')

    #Add final predictor for the current week of RZSM within reforecast
    idx+=1
    soil_var = 'soilw_bgrnd'
    channel_name_output = pred.return_channel_name(soil_var)
    channel_list.append(f'{channel_name_output}_ref_lead{lead}')

    print(f'And finally adding RZSM week {lead} from reforecast as a predictor')
    
    training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx, ref_source,final_testing_year)

    print(f'Index idx value is {idx}. Done adding {soil_var} reforecast.')

    print('Final channel of input is the following (just make sure it has zeros for RZSM)')
    print(training_input[0,:,:,-1])

    return(training_input, validation_input, testing_input,channel_list)


def load_only_predictions_EX26(lead, input_directory,training_size_shape,validation_testing_size_shape,RZSM_or_Tmax_or_both,experiment_name,region_name,
                               experiment_name_out,ref_source):
    '''We need to combine all the RZSM files and all the observation data (pwat, spfh, tmax, diff_temp, z200) into one file. Also adding the RZSM reforecast data from RZSM and Tmax '''

    channel_list = []

    assert lead >=2, 'We cannot make a prediction if lead week is 1 because we dont have an input. So make lead week >= 2.'

    #Get the number of extra channels to add from previous weeks predictions
    add_channels = lead # We are only adding the previous weeks lag and the current week. 

    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],add_channels))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],add_channels))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],add_channels))
    
    print(f'Training input shape = {training_input.shape}')
    
    #For masking
    RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory)

    ############################### NOW ADD THE PREDICTION DATA FROM PREVIOUS WEEK ################################################
        #Now we need to load our data from the previous weeks
    for idx,lead_previous in enumerate(range(1,lead)):
        # break
        
        pred_dir = f'predictions/{region_name}/Wk{lead_previous}_training_validation'
        testing_dir = f'predictions/{region_name}/Wk{lead_previous}_testing'
        
        os.system(f'mkdir -p {pred_dir}')
        
        previous_model = return_fully_autoregressive_EX26(lead)
            
        
        training_prediction_name= f'{pred_dir}/Wk{lead_previous}_training_{previous_model[idx]}.npy'
        validation_prediction_name = f'{pred_dir}/Wk{lead_previous}_validation_{previous_model[idx]}.npy'
        testing_prediction_name = f'{testing_dir}/Wk{lead_previous}_testing_{previous_model[idx]}.npy'
        
        testing_prediction = np.load(testing_prediction_name)
        '''Load the training and validation predictions if they exist'''
        
        if os.path.exists(training_prediction_name):
            print(f'Loading data from previous week lead {lead_previous}')
            training_prediction = np.load(training_prediction_name)
            validation_prediction =np.load(validation_prediction_name)
            
        else:
            print(f'Creating prediction data from previous week lead {lead_previous}')
            '''If they don't exist..... load model for previous week'''
            model = load_model(f'checkpoints/Wk_{lead_previous}/Wk{lead_previous}_{experiment_name}',compile=False) #don't need the custom loss function for predictions
            
            training_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_training_input.npy')
            validation_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_validation_input.npy')
            testing_input_previous = np.load(f'Data/model_npy_inputs{region_name}//Wk{lead_previous}_EX_input_data/{experiment_name_out}_testing_input.npy')
            
            '''Make predictions and save data to load for previous use'''
            training_prediction = model.predict(training_input_previous) #Just for unit checking
            np.save(training_prediction_name,training_prediction)
            
            validation_prediction = model.predict(validation_input_previous)
            np.save(validation_prediction_name,validation_prediction)
        
        
        ############# ADD BACK TO CREATED ARRAY ###############################
        '''Now get the best UNET index prediction for RZSM and Tmax. Because the UNET plus plus produces 4 different outputs'''
        min_RZSM_index,min_tmax_index = f.load_loss_csv_bias_correction(lead_previous,previous_model[idx],RZSM_or_Tmax_or_both, region_name)

        if np.array(training_prediction).shape[0] == 8 and RZSM_or_Tmax_or_both == 'both':
            min_tmax_index = min_tmax_index+4 #Because index values 0-3 are RZSM and 4-7 are Tmax
        elif np.array(training_prediction).shape[0] == 6 and RZSM_or_Tmax_or_both == 'both':
            min_tmax_index = min_tmax_index + 3 #Because index values are 0-2 for RZSM and 3-5 for Tmax
        '''Now split the training and validation data back into RZSM and/or Tmax'''
        
        
        train_RZSM = np.array(training_prediction)[min_RZSM_index,:,:,:,:]
        val_RZSM = np.array(validation_prediction)[min_RZSM_index,:,:,:,:]
        test_RZSM = np.array(testing_prediction)[min_RZSM_index,:,:,:,:]


        '''Make sure we mask RZSM values which aren't on land'''
        train_RZSM = np.where(RZSM_train_obs==0,0,train_RZSM)
        val_RZSM = np.where(RZSM_validation_obs==0,0,val_RZSM)
        test_RZSM = np.where(RZSM_validation_obs==0,0,test_RZSM)

        #Now add back to the data
        '''Now add back to the newly created dataset'''
        channel_list.append(f'RZSM_prediction_lead{lead_previous}')
        print(f'Adding RZSM training, validation, testing into index {idx}')
        training_input[:,:,:,idx] = train_RZSM[:,:,:,0]
        validation_input[:,:,:,idx] = val_RZSM[:,:,:,0]
        testing_input[:,:,:,idx] = test_RZSM[:,:,:,0]

        idx+=1

    
    ############################### NOW ADD THE REFORECAST DATA FROM THE WEEK TO BE PREDICTED ################################################
    
    #Add RZSM reforecast week N
    soil_var = 'soilw_bgrnd'
    channel_name_output = pred.return_channel_name(soil_var)
    channel_list.append(f'{channel_name_output}_ref_lead{lead}')

    print(f'And finally adding RZSM week {lead} from reforecast as a predictor')
    
    training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory,soil_var, idx)

    print(f'Index idx value is {idx}. Done adding {soil_var} reforecast.')

    print('Final channel of input is the following (just make sure it has zeros for RZSM)')
    print(training_input[0,:,:,-1])

    return(training_input, validation_input, testing_input,channel_list)



def load_all_data_EX29_after_week_0(lead, 
                                    num_lags_obs_RZSM, 
                                    include_lags_obs_pwat_spfh_tmax, 
                                    include_reforecast_or_not,
                                    observation_lag_list_not_RZSM, 
                                    lag_integer_list, 
                                    input_directory,
                                    training_size_shape,
                                    validation_testing_size_shape,
                                    experiment_name,
                                    RZSM_or_Tmax_or_both, 
                                    experiment_test,
                                    region_name,
                                    experiment_name_out,
                                    obs_source,
                                    ref_source,
                                   final_testing_year):
    '''We need to combine all the RZSM files and all the observation data (pwat, spfh, tmax, diff_temp, z200) into one file. Also adding the RZSM reforecast data from RZSM and Tmax '''

    channel_list = []
    
    #Reforecast predictors
    var_list = ['pwat_eatm', 'spfh_2m', 'tmax_2m', 'diff_temp_2m', 'hgt_pres']
    
    if ref_source == 'GEFSv12':
        var_list_ref = var_list
    elif ref_source == 'ECMWF':
        var_list_ref = ['t2m', 'd2m', 'tcw']
        
    if (experiment_test == 2) and (lead ==1):
        #This is adding 5 additional predictors (tmax, diff_temp, z200, pwat, spfh)
        #Also add the number of channels according to the lead. If lead == 1, then only add lead 1 current week. If lead == 2, add one channel for week 2 and one channel for the prediction from Week 1
        add_channels = len(lag_integer_list)+len(var_list)+len(var_list_ref)  #Only add a single prediction week from these
        reforecast_predictors = True
    elif (experiment_test == 2) and (lead ==2):
        add_channels = len(lag_integer_list) + len(var_list)+len(var_list_ref) + (lead -1) 
        reforecast_predictors = True   
    elif (experiment_test == 2) and (lead > 2):
        add_channels = len(lag_integer_list) + (lead -1) 
        reforecast_predictors = False  
        
    training_input = np.empty(shape = (training_size_shape[0],training_size_shape[1],training_size_shape[2],add_channels))
    validation_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],add_channels))
    testing_input = np.empty(shape = (validation_testing_size_shape[0],validation_testing_size_shape[1],validation_testing_size_shape[2],add_channels))
    
    print(f'Training input shape = {training_input.shape}')
    
    #Add RZSM observations first
    for idx,lag in enumerate(lag_integer_list):
        channel_list.append(f'RZSM_obs_lag{lag}')
        training_input, validation_input, testing_input = pred.add_obs_RZSM_by_lag(training_input, validation_input, testing_input, lag, input_directory, idx, final_testing_year, obs_source)
        
    print(f'Index idx value is {idx}. Done adding RZSM obs.')    
    
    RZSM_train_obs, RZSM_validation_obs = pred.return_masking_objects_for_RZSM(input_directory, final_testing_year,obs_source)

    if lead <=2:
        for variable in var_list:
            channel_name_output = pred.return_channel_name(variable)
            for lag in [-1]:
                '''Only include the first lag'''
                channel_list.append(f'{channel_name_output}_obs_lag{lag}')
                idx+=1
                #Observations adding
                training_input, validation_input, testing_input = pred.add_obs_other_observations_by_lag(training_input, validation_input, testing_input, lag, input_directory, variable, idx,final_testing_year)
     
        
            print(f'Index idx value is {idx}. Done adding {variable} obs.')
        
    ############################### NOW ADD THE PREDICTION DATA FROM PREVIOUS WEEK ################################################
    #Now we need to load our data from the previous weeks 
    
    if lead in [0, 1]:
        #DO NOT INCLUDE LEAD 0 AS A PREDICTOR (FROM PREVIOUS WEEK)
        pass
    else:

        for lead_previous in range(1,lead):
            print(f'Adding previous RZSM prediction from week {lead_previous} as an input channel')
            # break
            
            pred_dir = f'predictions/{region_name}/Wk{lead_previous}_training_validation'
            testing_dir = f'predictions/{region_name}/Wk{lead_previous}_testing'
            
            os.system(f'mkdir -p {pred_dir}')
            
            training_prediction_name= f'{pred_dir}/Wk{lead_previous}_training_{experiment_name_out}.npy'
            validation_prediction_name = f'{pred_dir}/Wk{lead_previous}_validation_{experiment_name_out}.npy'
            testing_prediction_name = f'{testing_dir}/Wk{lead_previous}_testing_{experiment_name_out}.npy'
            
            testing_prediction = np.load(testing_prediction_name)
            
            '''Load the training and validation predictions if they exist'''
            if os.path.exists(training_prediction_name):
                print(f'Loading data from previous week lead {lead_previous}')
                training_prediction = np.load(training_prediction_name)
                validation_prediction =np.load(validation_prediction_name)
                
            else:
                print(f'Creating prediction data from previous week lead {lead_previous}')
                '''If they don't exist..... load model for previous week'''
                model = load_model(f'checkpoints/Wk_{lead_previous}/Wk{lead_previous}_{experiment_name_out}',compile=False) #don't need the custom loss function for predictions
                
                training_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_training_input.npy')
                validation_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_validation_input.npy')
                testing_input_previous = np.load(f'Data/model_npy_inputs/{region_name}/Wk{lead_previous}_EX_input_data/{experiment_name_out}_testing_input.npy')
                
                '''Make predictions and save data to load for previous use'''
                training_prediction = model.predict(training_input_previous) #Just for unit checking
                np.save(training_prediction_name,training_prediction)
                
                validation_prediction = model.predict(validation_input_previous)
                np.save(validation_prediction_name,validation_prediction)
            
            
            ############# ADD BACK TO CREATED ARRAY ###############################
            '''Now get the best UNET index prediction for RZSM and Tmax. Because the UNET plus plus produces 4 different outputs'''
            min_RZSM_index,min_tmax_index = f.load_loss_csv_bias_correction(lead_previous,experiment_name_out,RZSM_or_Tmax_or_both,region_name)
    
            if np.array(training_prediction).shape[0] == 8 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index+4 #Because index values 0-3 are RZSM and 4-7 are Tmax
            elif np.array(training_prediction).shape[0] == 6 and RZSM_or_Tmax_or_both == 'both':
                min_tmax_index = min_tmax_index + 3 #Because index values are 0-2 for RZSM and 3-5 for Tmax
            '''Now split the training and validation data back into RZSM and/or Tmax'''
            
            
            train_RZSM = np.array(training_prediction)[min_RZSM_index,:,:,:,:]
            val_RZSM = np.array(validation_prediction)[min_RZSM_index,:,:,:,:]
            test_RZSM = np.array(testing_prediction)[min_RZSM_index,:,:,:,:]

    
            '''Make sure we mask RZSM values which aren't on land'''
            train_RZSM = np.where(RZSM_train_obs==0,0,train_RZSM)
            val_RZSM = np.where(RZSM_validation_obs==0,0,val_RZSM)
            test_RZSM = np.where(RZSM_validation_obs==0,0,test_RZSM)
    
            #Now add back to the data
            '''Now add back to the newly created dataset'''
            idx+=1
            channel_list.append(f'RZSM_prediction_lead{lead_previous}')
            print(f'Adding RZSM training, validation, testing into index {idx}')
            training_input[:,:,:,idx] = train_RZSM[:,:,:,0]
            validation_input[:,:,:,idx] = val_RZSM[:,:,:,0]
            testing_input[:,:,:,idx] = test_RZSM[:,:,:,0]

    
    ############################### NOW ADD THE REFORECAST DATA FROM THE WEEK TO BE PREDICTED ################################################
    
    #Add predictors from the reforecast model
    if (lead in [1,2]):
        print(f'Adding current reforecast data from week {lead} for vars {(var_list_ref)}')  
        #Only add these for week 1 and 2. 
        for variable in var_list_ref:
            idx+=1
            channel_name_output = pred.return_channel_name(variable)
            
            channel_list.append(f'{channel_name_output}_ref_lead{lead}')
            
            training_input, validation_input, testing_input = pred.add_reforecast_by_lag(training_input, validation_input, testing_input, lead, input_directory, variable, idx, ref_source, final_testing_year)
    
            print(f'Index idx value is {idx}. Done adding {variable} reforecast.')

    print(training_input[0,:,:,-1])

    return(training_input, validation_input, testing_input,channel_list)



