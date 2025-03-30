library(ensembleMOS)
library(vroom)
library(dplyr)
library(data.table)
library(parallel)
#https://search.r-project.org/CRAN/refmans/ensembleMOS/html/ensembleMOS.html

# load('Rworkspace_emos7.Rdata')

# # # # #test setup
# lead_day=as.double(7) #added by bash script
# data = fread(file=paste0('emos_setup_train_val_test_lead',lead_day,'.csv.gz'))
# #get the information about all grid cells
# data = data.frame(data)
# 
# #remove np.nan grid cells
# data=na.omit(data)
# 
# all_grid_cell_num <- length(rownames(unique(data['lat_lon'])))
# all_grid_cells<- unique(data['lat_lon'])
# grid_number=1
# training_weeks=831
# #week 1 emos
# final_dataframe_leadN=data.frame()

# Lead1 EMOS --------------------------------------------------------------
# bad_data<<-0


emos_function_all_days_skip_bad_dates <- function(grid_number){
  # Partition test data
  # test_subset  <- data %>% filter(vdate > '2017010100')
  # train_subset <- data %>% filter(vdate < '2018-01-01')
  #Need to find the first date that is 12 weeks before the start of the training period
  
  #From manual inspection, we can see that 2017-10-04 is 13 weeks before the start of our testing dataset (2018-2019)
  # test_subset  <- test_subset %>% filter(vdate >= '2017100400')
  
  if (lead_day==0){
    lead_increment=1
  }else {
    lead_increment=lead_day
  }
  # Partition test data
  test_subset  <- data %>% filter(vdate > '2018010100')
  count_validation_days_test <- row(unique(test_subset['vdate'])) #get number of days in the the testing dataset
  validation_days_test <- unique(test_subset['vdate']) #get the actual dates
  string_validation_days_test <- as.character(validation_days_test[,1]) #convert the dates to a string for later use
  
  
  #Partition training data
  train_subset  <- data %>% filter(vdate < '2016010100')
  count_validation_days_train <- row(unique(train_subset['vdate'])) #get number of days in the the testing dataset
  validation_days_train <- unique(train_subset['vdate']) #get the actual dates
  string_validation_days_train <- as.character(validation_days_train[,1]) #convert the dates to a string for later use
  num_training_days <- length(unique(validation_days_train)[,])
  
  training_weeks <- num_training_days
  
  
  #must not have dates before training period starts, so choose 1 week after the 3 month (or 1 week after # training days which is actually weeks for my research)
  # string_validation_days <-string_validation_days[training_weeks+(lead_day/7):length(string_validation_days)] 
  # num_training <- length(validation_days)
  
  #save the dates of initialization
  init_days = unique(test_subset['idate'])[,1]
  # init_days=init_days[training_weeks+(lead_day/7):length(init_days)] 
  # init_days=init_days[2:102]
  
  obs <- paste("RZSM", "obs", sep = ".")
  ensMemNames <- c("RZSM.0","RZSM.1","RZSM.2","RZSM.3","RZSM.4","RZSM.5","RZSM.6",
                   "RZSM.7","RZSM.8","RZSM.9","RZSM.10")
  
  #This algorithm can only do 1 location at a time.
  #testing with 1 location:
  lat_longitude <- all_grid_cells[grid_number,] #get lat_lon actual value
  single_grid_cell_test <- subset(test_subset,lat_lon == lat_longitude) #Subset only that specific lat_lon value
  single_grid_cell_train <- subset(train_subset,lat_lon == lat_longitude)
  rownames(single_grid_cell_test) <- 1:nrow(single_grid_cell_test)  #just renumber the rows to be consistent
  rownames(single_grid_cell_train) <- 1:nrow(single_grid_cell_train)  #just renumber the rows to be consistent
  
  #Just replace the np.nan values because they will mess up the algorithm
  single_grid_cell_train$RZSM.10[is.na(single_grid_cell_train$RZSM.10)]<-mean(single_grid_cell_train$RZSM.10,na.rm=TRUE)
  single_grid_cell_train$RZSM.9[is.na(single_grid_cell_train$RZSM.9)]<-mean(single_grid_cell_train$RZSM.9,na.rm=TRUE)
  single_grid_cell_train$RZSM.8[is.na(single_grid_cell_train$RZSM.8)]<-mean(single_grid_cell_train$RZSM.8,na.rm=TRUE)
  single_grid_cell_train$RZSM.7[is.na(single_grid_cell_train$RZSM.7)]<-mean(single_grid_cell_train$RZSM.7,na.rm=TRUE)
  single_grid_cell_train$RZSM.6[is.na(single_grid_cell_train$RZSM.6)]<-mean(single_grid_cell_train$RZSM.6,na.rm=TRUE)
  single_grid_cell_train$RZSM.5[is.na(single_grid_cell_train$RZSM.5)]<-mean(single_grid_cell_train$RZSM.5,na.rm=TRUE)
  single_grid_cell_train$RZSM.4[is.na(single_grid_cell_train$RZSM.4)]<-mean(single_grid_cell_train$RZSM.4,na.rm=TRUE)
  single_grid_cell_train$RZSM.3[is.na(single_grid_cell_train$RZSM.3)]<-mean(single_grid_cell_train$RZSM.3,na.rm=TRUE)
  single_grid_cell_train$RZSM.2[is.na(single_grid_cell_train$RZSM.2)]<-mean(single_grid_cell_train$RZSM.2,na.rm=TRUE)
  single_grid_cell_train$RZSM.1[is.na(single_grid_cell_train$RZSM.1)]<-mean(single_grid_cell_train$RZSM.1,na.rm=TRUE)
  single_grid_cell_train$RZSM.0[is.na(single_grid_cell_train$RZSM.0)]<-mean(single_grid_cell_train$RZSM.0,na.rm=TRUE)
  
  
  
  data_setup <- function(date_){
    
    combined_train_and_test <- rbind(single_grid_cell_train, single_grid_cell_test[date_,])
    init_date <- single_grid_cell_test[date_,'idate']
    training_leadN <- ensembleData(forecasts = combined_train_and_test[,ensMemNames],
                                   dates = combined_train_and_test[,"vdate"],
                                   observations = combined_train_and_test[,obs],
                                   forecastHour = 24*lead_day,
                                   initializationTime = "00",
                                   consecutive=FALSE)
    
    return(list('combined_train_and_test' = combined_train_and_test, 'init_date' = init_date, 'training_leadN' = training_leadN))
    }
  
  #Now combine the training and testing dataset into a single timeseries
  #We must loop through each one because otherwise it will create an unfair advantage between training and testing data
  #We must add a 2 because otherwise we are going to get some data that was initialized from 12/27/2017
  for (date_ in 2:length(string_validation_days_test)){
    # print(paste0('Working on date ', string_validation_days_test[date]))
    # We only need to look at 1 at a time. We can't include any test data for training purposes
    
    data_setup_output <- data_setup(date_)
    
    # return_info$combined_train_and_test

  #normally forecast hour would be number of hours between idate and vdate, but
  #I don't think it's working like the same script does. It models dates outside of
  #the training set
  
  #sometimes there are no values for the observations (because of some re-gridding)
  if (sum(is.na(data_setup_output$training_leadN$observations)) ==length(single_grid_cell_test$RZSM.obs)) {
    output <- data.frame(emos_mean_prediction = NA,
                         emos_mean_std=NA,
                         lat_longitude,
                         init=actual_init_days,
                         observation_day=string_validation_days_test,
                         observation=NA,
                         ensemble_crps=NA,
                         emos_crps=NA,
                         mae=NA,
                         mse=NA,
                         rmse=NA)
    
    #add that day back to the dataframe
    final_dataframe_leadN=rbind(final_dataframe_leadN,output)
    
    
    
  }else{
  

      #Sometimes there is a date that just doens't work (not sure why) so let's do a try catch statement and just skip over it
      tempTestFit <- ensembleMOS(data_setup_output$training_leadN, trainingDays = training_weeks, 
                                 model = "normal",dates = string_validation_days_test[date_])
      # crpsValues <- colMeans(crps(tempTestFit, training_leadN)) #first column = CRPS of raw ensemble, 2nd column for EMOS
      crpsValues <- crps(tempTestFit, data_setup_output$training_leadN) #first column = CRPS of raw ensemble, 2nd column for EMOS
      # cdfValues <- cdf(tempTestFit, training_leadN,values=seq(from=0,to=1,by=0.01))
      parValues <- pars(tempTestFit, data_setup_output$training_leadN) #mean and standard deviation from forecast ensembles
      #based on the multiple linear regression equation, get the predicted mean (this is actually the same answer from the pars function!!!)
      # prediction <- sum(tempTestFit$B[,1] * training_leadN[14,1:11]) +tempTestFit$a[1] 
      # quantiles <- quantileForecast(tempTestFit,training_leadN,c(seq(0,1,by=0.1)))
      
      #Need to find the observations, select 12+lead_day/7 to account for different leads
      #EMOS doesn't have a good structure and doesn't keep up with the days that the forecasts were made,
      #we must be really careful about this
      observations <- single_grid_cell_test[date_,'RZSM.obs']
      length(rownames(single_grid_cell_test))
      # length(observations)
      # length(abs(parValues[,1]))
      # length(crpsValues[,1])
      
      mae <- abs(parValues[,1] - observations)
      mse <- (parValues[,1] - observations)**2
      rmse <- sqrt(mse)
      
      
      #save a dataframe with necessary data
      output <- data.frame(emos_mean_prediction = parValues[,1],
                           emos_mean_std=parValues[,2],
                           lat_longitude,
                           init=data_setup_output$init_date,
                           observation_day=single_grid_cell_test[date_,'vdate'],
                           observation=observations,
                           ensemble_crps=crpsValues[,1],
                           emos_crps=crpsValues[,2],
                           mae=mae,
                           mse=mse,
                           rmse=rmse)
      
      final_dataframe_leadN=rbind(final_dataframe_leadN,output)
  }
}
  return(final_dataframe_leadN) #for the non-data init dates/grid cells
}





#week 1 emos
lead_day<<-as.double('21') #added by bash script
# lead_day=as.double('7')
# data = read.csv(file=paste0('emos_setup_lead',lead_day,'.csv'))
#get the information about all grid cells
data <- fread(file=paste0('emos_setup_train_val_test_lead',lead_day,'.csv.gz'))
data <- data.frame(data)

#remove np.nan grid cells
data<-na.omit(data)

all_grid_cell_num <<- length(rownames(unique(data['lat_lon'])))
all_grid_cells<<- unique(data['lat_lon'])


final_dataframe_leadN<<-data.frame()

numberOfCores <- 12

final_output <-do.call(
  rbind,mclapply(1:all_grid_cell_num,emos_function_all_days_skip_bad_dates,mc.cores=numberOfCores))
  
  
write.csv(final_output,paste0('emos_completed_lead',lead_day,'_test_predictions_only_from_training_dataset.csv'),row.names = FALSE)


save.image(paste0('Rworkspace_emos',lead_day,'test_predictions_only_from_training_dataset.Rdata'))

