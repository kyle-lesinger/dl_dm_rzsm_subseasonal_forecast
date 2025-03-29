import numpy as np
import tensorflow as tf
from keras import backend as K
from function import denseValue

# global density
# # density = np.load('Data/model_npy_inputs/weighted_density/weighted_density_0.8.npy')
# density = np.load('Data/density.npy')
# density = tf.cast(np.nan_to_num(density, nan=0),dtype=tf.float32)


## Source https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/losses.py
def _crps_tf(y_true, y_pred, factor=0.08):
    
    '''
    core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''
    
    # mean absolute error
    mae = tf.cast(K.mean(tf.abs(y_pred - y_true)),dtype=tf.float32)
    
    # dist = tf.math.reduce_std(y_pred) #places the same results as numpy apply to axis 1 then taking the mean over all grid cells (roughly)
    
    np_ypred = y_pred.numpy()
    dist = np.nanmean(np.nanstd(np_ypred,axis=0))
    dist = tf.cast(dist,dtype=tf.float32)
    
    return mae - factor*dist

def crps2d_tf(y_true, y_pred, factor=0.08):
    
    '''
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level Keras training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        
    '''
    # tf.print(f'y_true shape = {y_true.shape}')
    # tf.print(f'y_pred shape = {y_pred.shape}')
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)
    # batch_num = y_pred.shape.as_list()[0]
    
    crps_out = 0
    start_,end_ = 0,11
    
    #Split the batch sizes to not get an Nan at the last part of the batch
    new_range = y_true.shape[0]//11
    for i in range(new_range):
        crps_out += _crps_tf(y_true=y_true[start_:end_, ...], y_pred =y_pred[start_:end_, ...], factor=factor)
        #must add 11 each time
        start_ +=11
        end_ += 11
    
    #Divide by total number of inits
    crps_out = crps_out/new_range
        
    return crps_out


## Source https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/losses.py
def _crps_tf_dense(y_true, y_pred, factor=0.08):
    
    '''
    core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)

    #Maybe experiment with having the kde*MAE
    def mae_by_whole_region_mean(y_true, y_pred, factor=0.08):
        #MAE by grid cell then averaged over all study region
        mae = tf.cast(K.mean(tf.abs(y_pred - y_true)),dtype=tf.float32)
        dist = tf.cast(np.nanmean(np.nanstd(y_pred,axis=0)),dtype=tf.float32)
        crps_exp = mae - factor*dist
    
        #Just provide the names so we know what they are
        norm_obs = density[:,:,:,0] #Shape is (5844, 48, 96)
        norm_kde = density[:,:,:,1] #Shape is (5844, 48, 96)
    
        #y_pred shape is TensorShape([11, 48, 96])
        y_pred_mean = tf.experimental.numpy.nanmean(y_pred,axis=0)
        #Try with just taking the mean of the predictions
        #Find the absolute value difference which is equal to us finding which observation value is closest to the prediction value (y_pred)
        # y_pred shape is (48,96)
        abs_diff = tf.math.abs(norm_obs - tf.experimental.numpy.nanmean(y_pred,axis=0)) # Shape is TensorShape([5844, 48, 96])
        
        #Find the date index that is the lowest
        min_day_indices = tf.math.argmin(abs_diff,axis=0) #Shape is (48,96)
    
        #Check some values (looks good)
        # x,y=10,20
        # min_arg = argmin[x,y]
        # abs_diff
        # norm_obs[min_arg,x,y]
        # y_pred_mean[x,y]
    
        #Now we need to subset norm_kde to find its value
        # Use tf.gather to obtain the subset of 'norm_kde'
        subset_array = tf.gather(norm_kde, min_day_indices[:, tf.newaxis, tf.newaxis], axis=0)
        # Squeeze the tensor to remove singleton dimensions
        avg_kde = K.mean(tf.squeeze(subset_array, axis=[1, 2]))
    
    
        
        #Now multiply the grid with the data and take the mean to produce a scalar
        dense_crps = crps_exp * avg_kde

        # #check the other result where we apply the mean first to produce a scalar with no dense crps
        # mae = tf.cast(K.mean(tf.abs(y_pred - y_true)),dtype=tf.float32)
        # #Then find another scalar
        # dist = np.nanmean(np.nanstd(y_pred,axis=0))
        # dist = tf.cast(dist,dtype=tf.float32)
        # #Then compute a scalar
        # (mae - factor*dist)
        
        return(dense_crps)



    def mae_by_grid_cell_then_take_mean(y_true, y_pred, factor=0.08):
        #right now this is too difficult to try and get to work performance wise, So it's currently unfinished. 
        #MAE by grid cell
        mae = tf.abs(y_pred - y_true)
        dist = np.nanstd(y_pred,axis=0)
        crps_exp = mae - factor*dist
    
        #Just provide the names so we know what they are
        norm_obs = density[:,:,:,0] #Shape is (5844, 48, 96)
        norm_kde = density[:,:,:,1] #Shape is (5844, 48, 96)
    
        

        #Find the absolute value difference which is equal to us finding which observation value is closest to the prediction value (y_pred)
        # y_pred shape is (48,96)
        abs_diff = tf.math.abs(norm_obs -y_pred[:,tf.newaxis,:,:]) # Shape is TensorShape([11, 5844, 48, 96])
        
        #Find the date index that is the lowest
        min_day_indices = tf.math.argmin(abs_diff,axis=1) #Shape is (11,48,96)
    
        #Check some values (looks good)
        # x,y=10,20
        # min_arg = argmin[x,y]
        # abs_diff
        # norm_obs[min_arg,x,y]
        # y_pred_mean = tf.experimental.numpy.nanmean(y_pred,axis=0) #y_pred shape is TensorShape([48, 96])
        # y_pred_mean[x,y]
    
        #Now we need to subset norm_kde to find its value
        # Use tf.gather to obtain the subset of 'norm_kde'
        subset_array = tf.gather(norm_kde, min_day_indices[tf.newaxis, :], axis=0)
        subset_array.shape
        # Squeeze the tensor to remove singleton dimensions
        avg_kde = K.mean(tf.squeeze(subset_array, axis=[1, 2]))

        #Now multiply the grid with the data and take the mean to produce a scalar
        dense_crps = crps_exp * avg_kde
        return(dense_crps)

    return mae_by_whole_region_mean(y_true, y_pred, factor=0.08)

def crps2d_tf_dense(y_true, y_pred, factor=0.08):
    
    '''

    #But in this test, we are computing experimental CRPS and then multiplying by the KDE dense loss function

    
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level Keras training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        
    '''
    # tf.print(f'y_true shape = {y_true.shape}')
    # tf.print(f'y_pred shape = {y_pred.shape}')
    # tf.print('PREDICTION')
    # tf.print(y_pred[0,:,:,0])
    # tf.print('TRUE')
    # tf.print(y_true[0,:,:,0])
    
    # # y_pred = np.where(y_true != 0, y_pred, 0)
    # y_true_zeros = y_true[y_true==0]
    # y_pred[y_true_zeros] = 0
    
    # tf.print('PREDICTION MASKED')
    # tf.print(y_pred[0,:,:,0])
    
    # Create a mask for indices where observation is 0
    mask = tf.equal(y_true, 0)

    # Apply the mask to reforecasts
    y_pred = tf.where(mask, 0, y_pred)
        
    y_pred = tf.squeeze(tf.convert_to_tensor(y_pred))
    y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))

    #For testing
    # y_pred = y_pred[0:66,:,:]
    # y_true = y_true[0:66,:,:]
    
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)
    # batch_num = y_pred.shape.as_list()[0]
    
    crps_out = 0
    start_,end_ = 0,11

    #testing
    #factor=0.08
    
    #Split the batch sizes to not get an Nan at the last part of the batch
    new_range = y_true.shape[0]//11
    for i in range(new_range):
        # print(i)
        # break
        crps_out += _crps_tf_dense(y_true=y_true[start_:end_, ...], y_pred =y_pred[start_:end_, ...], factor=factor)
        #must add 11 each time for the 11 differenct ensemble members
        start_ +=11
        end_ += 11
    
    #Divide by total number of inits
    crps_out = crps_out/new_range

    #Have them all as a sum
    #crps_out = crps_out
        
    return crps_out


## Source https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/losses.py
def _crps_tf_dense_test2(y_true, y_pred, factor=0.08):
    
    '''
    core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)

    #Maybe experiment with having the kde*MAE
    def mae_by_whole_region_mean(y_true, y_pred, factor=0.08):
        #MAE by grid cell then averaged over all study region
        mae = tf.cast(K.mean(tf.abs(y_pred - y_true)),dtype=tf.float32)
        dist = tf.cast(np.nanmean(np.nanstd(y_pred,axis=0)),dtype=tf.float32)
        
    
        #Just provide the names so we know what they are
        norm_obs = density[:,:,:,0] #Shape is (5844, 48, 96)
        norm_kde = density[:,:,:,1] #Shape is (5844, 48, 96)
    
        #y_pred shape is TensorShape([11, 48, 96])
        y_pred_mean = tf.experimental.numpy.nanmean(y_pred,axis=0)
        #Try with just taking the mean of the predictions
        #Find the absolute value difference which is equal to us finding which observation value is closest to the prediction value (y_pred)
        # y_pred shape is (48,96)
        abs_diff = tf.math.abs(norm_obs - tf.experimental.numpy.nanmean(y_pred,axis=0)) # Shape is TensorShape([5844, 48, 96])
        
        #Find the date index that is the lowest
        min_day_indices = tf.math.argmin(abs_diff,axis=0) #Shape is (48,96)
    
        #Check some values (looks good)
        # x,y=10,20
        # min_arg = argmin[x,y]
        # abs_diff
        # norm_obs[min_arg,x,y]
        # y_pred_mean[x,y]
    
        #Now we need to subset norm_kde to find its value
        # Use tf.gather to obtain the subset of 'norm_kde'
        subset_array = tf.gather(norm_kde, min_day_indices[:, tf.newaxis, tf.newaxis], axis=0)
        # Squeeze the tensor to remove singleton dimensions
        avg_kde = K.mean(tf.squeeze(subset_array, axis=[1, 2]))
    
    
        
        #Now multiply the grid with the data and take the mean to produce a scalar
        crps_exp = (mae*avg_kde) - (factor*dist)
        
        # #check the other result where we apply the mean first to produce a scalar with no dense crps
        # mae = tf.cast(K.mean(tf.abs(y_pred - y_true)),dtype=tf.float32)
        # #Then find another scalar
        # dist = np.nanmean(np.nanstd(y_pred,axis=0))
        # dist = tf.cast(dist,dtype=tf.float32)
        # #Then compute a scalar
        # (mae - factor*dist)
        
        return(crps_exp)



    def mae_by_grid_cell_then_take_mean(y_true, y_pred, factor=0.08):
        #right now this is too difficult to try and get to work performance wise, So it's currently unfinished. 
        #MAE by grid cell
        mae = tf.abs(y_pred - y_true)
        dist = np.nanstd(y_pred,axis=0)
        crps_exp = mae - factor*dist
    
        #Just provide the names so we know what they are
        norm_obs = density[:,:,:,0] #Shape is (5844, 48, 96)
        norm_kde = density[:,:,:,1] #Shape is (5844, 48, 96)
    
        

        #Find the absolute value difference which is equal to us finding which observation value is closest to the prediction value (y_pred)
        # y_pred shape is (48,96)
        abs_diff = tf.math.abs(norm_obs -y_pred[:,tf.newaxis,:,:]) # Shape is TensorShape([11, 5844, 48, 96])
        
        #Find the date index that is the lowest
        min_day_indices = tf.math.argmin(abs_diff,axis=1) #Shape is (11,48,96)
    
        #Check some values (looks good)
        # x,y=10,20
        # min_arg = argmin[x,y]
        # abs_diff
        # norm_obs[min_arg,x,y]
        # y_pred_mean = tf.experimental.numpy.nanmean(y_pred,axis=0) #y_pred shape is TensorShape([48, 96])
        # y_pred_mean[x,y]
    
        #Now we need to subset norm_kde to find its value
        # Use tf.gather to obtain the subset of 'norm_kde'
        subset_array = tf.gather(norm_kde, min_day_indices[tf.newaxis, :], axis=0)
        subset_array.shape
        # Squeeze the tensor to remove singleton dimensions
        avg_kde = K.mean(tf.squeeze(subset_array, axis=[1, 2]))

        #Now multiply the grid with the data and take the mean to produce a scalar
        dense_crps = crps_exp * avg_kde
        return(dense_crps)

    return mae_by_whole_region_mean(y_true, y_pred, factor=0.08)

def crps2d_tf_dense_test2(y_true, y_pred, factor=0.08):
    
    '''
    #But in this test, we are multiply the MAE by the average KDE value from the dense loss function

    
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level Keras training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        
    '''
    # tf.print(f'y_true shape = {y_true.shape}')
    # tf.print(f'y_pred shape = {y_pred.shape}')
    # tf.print('PREDICTION')
    # tf.print(y_pred[0,:,:,0])
    # tf.print('TRUE')
    # tf.print(y_true[0,:,:,0])
    
    # # y_pred = np.where(y_true != 0, y_pred, 0)
    # y_true_zeros = y_true[y_true==0]
    # y_pred[y_true_zeros] = 0
    
    # tf.print('PREDICTION MASKED')
    # tf.print(y_pred[0,:,:,0])
    
    # Create a mask for indices where observation is 0
    mask = tf.equal(y_true, 0)

    # Apply the mask to reforecasts
    y_pred = tf.where(mask, 0, y_pred)
        
    y_pred = tf.squeeze(tf.convert_to_tensor(y_pred))
    y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))

    #For testing
    # y_pred = y_pred[0:66,:,:]
    # y_true = y_true[0:66,:,:]
    
    # tf.print(y_true.shape)
    # tf.print(y_pred.shape)
    # batch_num = y_pred.shape.as_list()[0]
    
    crps_out = 0
    start_,end_ = 0,11

    #testing
    #factor=0.08
    
    #Split the batch sizes to not get an Nan at the last part of the batch
    new_range = y_true.shape[0]//11
    for i in range(new_range):
        # print(i)
        # break
        crps_out += _crps_tf_dense_test2(y_true=y_true[start_:end_, ...], y_pred =y_pred[start_:end_, ...], factor=factor)
        #must add 11 each time for the 11 differenct ensemble members
        start_ +=11
        end_ += 11
    
    #Divide by total number of inits
    crps_out = crps_out/new_range

    #Have them all as a sum
    #crps_out = crps_out
        
    return crps_out


def return_ECMWF_loss_name_and_function(testing_scenario):
    if testing_scenario == 'regular':
        '''This is the final one that I am sticking with. None of the other tests were significantly better'''
        
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = 'ECMWF_regular'
        loss_fn = crps2d_tf
    
    elif testing_scenario == 'dense':
        from function import modelRzsmDenseRelu as UNETRzsm
        
        save_loss_name = 'ECMWF_denseLoss'
        loss_fn = crps2d_tf_dense_test2
        
    elif testing_scenario == 'mse':
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = 'ECMWF_mse'
        loss_fn = 'mse'    
        
    elif testing_scenario == 'transformer':
        from function import modelRzsmReluTransformer as UNETRzsm
        
        save_loss_name = 'ECMWF_transformer'
        loss_fn = crps2d_tf_dense_test2
    
    elif testing_scenario == 'super_pixel':
        from function import modelRzsmReluPixelSuper as UNETRzsm
        
        save_loss_name = 'ECMWF_supPixel'
        loss_fn = crps2d_tf_dense_test2
    
    elif testing_scenario == 'attention':
        from function import modelRzsmReluAttention as UNETRzsm
        
        save_loss_name = 'ECMWF_attention'
        loss_fn = crps2d_tf_dense_test2
        
    elif testing_scenario == 'None':
        from function import modelRzsm2DdropoutRelu as UNETRzsm
        
        save_loss_name = 'ECMWF'
        loss_fn = crps2d_tf
    
    elif testing_scenario == 'denseLarge':
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = 'ECMWF_denseLarge'
        loss_fn = crps2d_tf_dense_test2
    return save_loss_name, loss_fn


def return_GEFSv12_loss_name_and_function(testing_scenario):
    if testing_scenario == 'dense':
        from function import modelRzsmDenseRelu as UNETRzsm
        
        save_loss_name = 'denseLoss'
        loss_fn = crps2d_tf_dense_test2
        
    elif testing_scenario == 'regular':
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = testing_scenario
        loss_fn = crps2d_tf
    elif testing_scenario == 'mse':
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = 'mse'
        loss_fn = 'mse'    
    elif testing_scenario == 'regularDepth':
        from function import modelRzsmReluExtraConvDepthwise as UNETRzsm
    
        save_loss_name = 'regularDepth'
        loss_fn = crps2d_tf
    
    elif testing_scenario == 'transformer':
        from function import modelRzsmReluTransformer as UNETRzsm
        
        save_loss_name = 'transformer'
        loss_fn = crps2d_tf_dense_test2
    
    elif testing_scenario == 'super_pixel':
        from function import modelRzsmReluPixelSuper as UNETRzsm
        
        save_loss_name = 'supPixel'
        loss_fn = crps2d_tf_dense_test2
    
    elif testing_scenario == 'attention':
        from function import modelRzsmReluAttention as UNETRzsm
        
        save_loss_name = 'attention'
        loss_fn = crps2d_tf_dense_test2
        
    elif testing_scenario == 'None':
        from function import modelRzsm2DdropoutRelu as UNETRzsm
        
        save_loss_name = ''
        loss_fn = crps2d_tf
    
    elif testing_scenario == 'denseLarge':
        from function import modelRzsmReluExtraConv as UNETRzsm
    
        save_loss_name = 'denseLarge'
        loss_fn = crps2d_tf_dense_test2
        
    elif testing_scenario == 'basic':
        from function import basicUNET as UNETRzsm
    
        save_loss_name = testing_scenario
        loss_fn = crps2d_tf
    
    elif testing_scenario == 'regularResidual':
        #For predicting residuals
        from function import modelRzsmReluExtraConvResidual as UNETRzsm
    
        save_loss_name = 'regularResidual'
        loss_fn = crps2d_tf
    return save_loss_name, loss_fn