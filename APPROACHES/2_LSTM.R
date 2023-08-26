##################################
## Approach 2: LSTM Forecasting ##
##################################


## Importing libraries
## -----------------------

library(dplyr)
library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)


## Configuring set-up (not always necessary)
## -----------------------

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}


## Defining all functions
## -----------------------


#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x {list}; List object containing the original incomplete time series
#'
main <- function(x){
  
}


#' blocking
#' 
#' Function to initialize the imputation process. Returns a data-frame with information regarding the "blocks"
#' in the time series (i.e., where we have continuous data and where we have missing values). Useful for 
#' creating the training data in upcoming steps. The function requires that the input series x: Does not
#' contain all NA values; does not have leading/ending NA values; is at least the length of window + forecast;
#' and that there is window + forecast data points before (after) the first (final) NA value in the series.
#' Will return a NULL block for any series that does not satisfy these requirements. 
#' @param x {list}; List object containing the original incomplete time series
#' @param window {int}; The number of time points in the past to consider when forecasting
#' @param forecast {int}; The number of time points to forecast
#' 
blocking <- function(x, window, forecast){
  
  ## Checking for validity of the series
  if (sum(is.na(x)) == length(x)){print('Entire series is NA.'); return(NULL)}
  if (is.na(x[1])){print('Remove leading NA values from the series.'); return(NULL)}
  if (is.na(x[length(x)])){print('Remove ending NA values from the series.'); return(NULL)}
  if (length(x) < (window+forecast)){print('Series does not meet length requirement.'); return(NULL)}
  if (sum(is.na(x[1:(window+forecast)])) > 0){print('Series requires more data points before first NA value.'); return(NULL)}
  if (sum(is.na(x[(length(x)-window):length(x)])) > 0){print('Series requires more data points after last NA value.'); return(NULL)}
  
  ## Defining parameters and initializing vector to store results
  N = length(x)
  block = c()
  condition = TRUE; i = 1; M = 0
  
  while(condition){
    
    ## When the current value is not null:
    if (!is.na(x[i])){
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (!is.na(x[i])){next}
        else{end_idx = i - 1; continue = FALSE}
      }
      M = M + 1; seq_info = c(start_idx, end_idx, end_idx - start_idx + 1, 1)
      block = c(block, seq_info)
    }
    
    ## When the current value is null:
    else{
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (is.na(x[i]) & i <= N){next}
        else{end_idx = i - 1; continue = FALSE}
      }
      M = M + 1; seq_info = c(start_idx, end_idx, end_idx - start_idx + 1, 0)
      block = c(block, seq_info)
    }
    if (i > N){condition = FALSE}
  }
  ## Formatting the final block
  block = as.data.frame(matrix(block, nrow = M, byrow = TRUE), dim = c(M, 4)) %>%
    dplyr::rename(Start = V1, End = V2, Gap = V3, Complete = V4)
  block$Complete = ifelse(block$Complete == 0, FALSE, TRUE)
  block$Valid = ifelse(block$Complete == FALSE, NA, 
                       ifelse(block$Gap >= (window + forecast), TRUE, FALSE))
  return(block)
}


#' series_to_X_Y
#' 
#' First, this function calls the blocking function to obtain the block information for the input
#' time series. Next, based on the desired window and forecast lengths, it produces training and 
#' testing data sets to be used for the neural network. 
#' @param x {list}; List object containing the original incomplete time series
#' @param window {int}; The number of time points in the past to consider when forecasting
#' @param forecast {int}; The number of time points to forecast
#'
series_to_X_Y <- function(x, window, forecast){
  
  ## Checking validity with blocking information
  block = blocking(x, window, forecast)
  if (is.null(block)){print('Invalid.'); return(NULL)}
  
  ## Initializing vectors to store data
  X_train = c(); X_test = c()
  Y_train = c()
  
  ## Defining helpful parameters
  train_indices = which(block$Valid == TRUE); n_train_indices = length(train_indices)
  test_indices = which(block$Complete == FALSE); n_test_indices = length(test_indices)
  M = length(x) - (n_train_indices * window) - sum(is.na(x))
  P = forecast - 1
  
  ## Generating training data
  for (idx in train_indices){
    
    N = block[idx,]$Gap
    start_idx = block[idx,]$Start
    end_idx = block[idx,]$End
    
    x_temp = x[start_idx:end_idx]
    
    for (i in 1:(N-window-P)){
      row = x_temp[i:(i+window-1)]
      label = x_temp[(i+window):(i+window+P)]
      X_train = c(X_train, row)
      Y_train = c(Y_train, label)
    }
  }
  X_train = array(matrix(X_train, nrow = M, byrow = TRUE), dim = c(M, window))
  Y_train = array(matrix(Y_train, nrow = M, byrow = TRUE), dim = c(M, forecast))
  
  ## Generating testing data
  if (length(test_indices) > 0){
    for (idx in test_indices){
      start_idx = block[idx,]$Start - window
      end_idx = block[idx,]$Start - 1
      X_test = c(X_test, x[start_idx:end_idx])
    }
    X_test = array(matrix(X_test, nrow = n_test_indices, byrow = TRUE), dim = c(n_test_indices, window, 1))
  }
  return(list(X_train, Y_train, X_test))
}


#' data_generator
#' 
#' This function is to be used to call series_to_X_Y which creates the training and testing data sets
#' for the input time series. This functions facilitates this proces for both a forward pass of the 
#' time series and a backwards pass. Once all data, is obtained the funcnction concatenates the 
#' forward and backward pass data together into a final data set. Lastly, we convert these arrays
#' into tensors for specific use in TensorFlow.
#' @param x {list}; List object containing the original incomplete time series
#' @param window {int}; The number of time points in the past to consider when forecasting
#' @param forecast {int}; The number of time points to forecast
#'
data_generator <- function(x, window, forecast){
  
  ## Forward pass data generation
  forward = series_to_X_Y(x, window, forecast)
  if (is.null(forward)){return(NULL)} ## Sanity check
  
  ## Backward pass data generation
  backward = series_to_X_Y(rev(x), window, forecast)
  if (is.null(backward)){return(NULL)} ## Sanity check
  
  ## Defining helpful parameters
  M = 2 * dim(forward[[1]])[1]
  
  ## Creating final data data matrices
  X_train = rbind(forward[[1]], backward[[1]])
  Y_train = rbind(forward[[2]], backward[[2]])
  X_test_for = forward[[3]]; X_test_back = backward[[3]]
  
  ## Fixing the formatting
  dim(X_train) = c(M, window, 1) 
  dim(Y_train) = c(M, forecast)
  
  ## Converting to tensors
  X_train = tf$constant(X_train)
  Y_train = tf$constant(Y_train)
  #if (!is.null(X_test_for)){X_test_for = tf$constant(X_test_for)}
  #if (!is.null(X_test_back)){X_test_back = tf$constant(X_test_back)}
  
  return(list(X_train, Y_train, X_test_for, X_test_back))
}


#' fit_model
#' 
#' This function is to be used to build, fit, and compile an LSTM neural network model using the
#' provided training data. Since we are using a rolling prediction window, model needs to be able
#' to predict on a single observation. Therefore, we fit a model using a specified batch size, 
#' extracted the fitted weights, and build a new model using the copied weights and a batch
#' size of one (which will allow us to create the rolling prediction window). This function 
#' returns the final single item model that will be used for predictions.
#' @param X_train {tensor}; Tensor containing the training input
#' @param Y_train {tensor}; Tensor containing the training targets
#' @param window {int}; The number of time points in the past to consider when forecasting
#' @param forecast {int}; The number of time points to forecast
#'
fit_model <- function(X_train, Y_train, window, forecast, batch, validation_split){
  
  ## Defining parameters
  EPOCHS = 60
  
  ## *********
  ## Need to be careful with:
  ## BATCH: Must be a factor of length(training_data)
  ## validation_split: batch sizes need to conform
  
  ## Building the model
  model = keras_model_sequential(name = 'Model') %>% 
    layer_lstm(64, stateful = FALSE, batch_input_shape = c(batch, window, 1)) %>%
    layer_dense(8, activation = 'relu') %>%
    layer_dense(1, activation = 'linear')
  
  ## Compiling 
  model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                    loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')
  ## Fitting
  model %>% fit(X_train, Y_train, shuflle = FALSE, validation_split = validation_split,
                epochs = EPOCHS, batch_size = batch, verbose = 0)
  
  ## Extracting fitted weights
  fitted_weights = model %>% get_weights()
  
  ## Rebuilding model for batch size of one
  single_item_model = keras_model_sequential(name = 'Model') %>% 
    layer_lstm(64, stateful = FALSE, batch_input_shape = c(1, window, 1)) %>%
    layer_dense(8, activation = 'relu') %>%
    layer_dense(1, activation = 'linear')
  
  ## Copying weights from fitted model
  single_item_model %>% set_weights(fitted_weights)
  
  ## Returning the newly compiled model
  single_item_model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                                loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')
  return(single_item_model)
}








## EXAMPLE 1: Single gap
## ---------------------

dev.off()
set.seed(10)
X = interpTools::simXt(N = 500, numFreq = 2, b = c(100, 200), numTrend = 0)$Xt
X_gapped = interpTools::simulateGaps(list(X), p = 0.1, g = 50, K = 1)
X0 = X_gapped[[1]]$p0.1$g50[[1]]

#X0 = X; X0[200:249] = NA
plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)


## Defining parameters
window = 5
forecast = 1

## Generating training and testing data
data = data_generator(X0, window = window, forecast = forecast)

X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]

dim(X_train)
dim(Y_train)
dim(X_test_forward)
dim(X_test_backward)

## Fitting the model on training data
model = fit_model(X_train, Y_train, window, forecast, batch = 20, validation_split = 0.25)

## Creating rolling prediction window
rolling_prediction <- function(model, x, X_test, window){
  
  ## Defining helpful parameters
  rolling_length = sum(is.na(x))
  preds = c()
  
  for (i in 1:rolling_length){
    
    ## Predicting on current input
    pred = model %>% predict(tf$constant(X_test), batch_size = 1, verbose = 0) %>% as.numeric()
    
    ## Appending results
    preds = c(preds, pred)
    
    ## Updating prediction series
    X_test = c(as.numeric(X_test), pred)[-1]
    dim(X_test) = c(1, window, 1)
  }
  return(preds)
}

forward_preds = rolling_prediction(model, X0, X_test_forward, window)
backward_preds = rev(rolling_prediction(model, X0, X_test_backward, window))

## Plotting the forward and backward predictions
plot_results <- function(forward_preds, backward_preds){
  par(mfrow = c(2, 1), oma = c(4, 4, 4, 4), mar = c(2, 2, 2, 1), xpd = FALSE)
  
  plot(375:475, X[375:475], type = 'l', lty = 3, ylim = c(-500, 500),
       xlab = NA, ylab = NA); grid()
  lines(375:475, X0[375:475], type = 'l', lwd = 2)
  lines(412:461, forward_preds, type = 'l', col = 'dodgerblue', lwd = 1.5)
  legend('topright', legend = c('Forward Pass'), col = c('dodgerblue'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  plot(375:475, X[375:475], type = 'l', lty = 3, ylim = c(-500, 500), 
       xlab = NA, ylab = NA); grid()
  lines(375:475, X0[375:475], type = 'l', lwd = 2)
  lines(412:461, backward_preds, type = 'l', col = 'red', lwd = 1.5)
  legend('topright', legend = c('Backward Pass'), col = c('red'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  title(main = 'LSTM Network Results', xlab = 'Time', ylab = 'X', 
        outer = TRUE, line = 0, cex.main = 2) 
}
plot_results(forward_preds, backward_preds)

## Computing prediction performance
X_for = X0; X_for[which(is.na(X0))] = forward_preds
X_back = X0; X_back[which(is.na(X0))] = backward_preds
X_hybrid = X0; X_hybrid[412:436] = X_for[412:436]; X_hybrid[437:461] = X_back[437:461]; 

#X_hybrid[which(is.na(X0))] = (forward_preds+backward_preds)/2
X_hwi = interpTools::parInterpolate(X_gapped, methods = c('HWI'))[[1]]$HWI$p0.1$g50[[1]]

## Plotting forward/backward hybrid predictions vs. HWI imputation
plot_results2 <- function(){
  par(mfrow = c(2, 1), oma = c(4, 4, 4, 4), mar = c(2, 2, 2, 1), xpd = FALSE)
  
  plot(375:475, X[375:475], type = 'l', lty = 3, ylim = c(-500, 500),
       xlab = NA, ylab = NA); grid()
  lines(375:475, X0[375:475], type = 'l', lwd = 2)
  lines(412:461, X_hybrid[412:461], type = 'l', col = 'darkorange', lwd = 1.5)
  legend('topright', legend = c('LSTM Hybrid'), col = c('darkorange'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  plot(375:475, X[375:475], type = 'l', lty = 3, ylim = c(-500, 500), 
       xlab = NA, ylab = NA); grid()
  lines(375:475, X0[375:475], type = 'l', lwd = 2)
  lines(412:461, X_hwi[412:461], type = 'l', col = 'forestgreen', lwd = 1.5)
  legend('topright', legend = c('HWI'), col = c('forestgreen'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  title(main = 'LSTM Network Results', xlab = 'Time', ylab = 'X', 
        outer = TRUE, line = 0, cex.main = 2) 
}
plot_results2()

## Computing performance metrics
print(paste0('LSTM RMSE (forward): ', round(sqrt(mean((X - X_for)^2)), 3)))
print(paste0('LSTM RMSE (backward): ', round(sqrt(mean((X - X_back)^2)), 3)))
print(paste0('LSTM RMSE (hybrid): ', round(sqrt(mean((X - X_hybrid)^2)), 3)))
print(paste0('HWI RMSE: ', round(sqrt(mean((X - X_hwi)^2)), 3)))

print(paste0('LSTM MAE: ', round(mean(abs(X - X_for)), 3)))
print(paste0('LSTM MAE: ', round(mean(abs(X - X_back)), 3)))
print(paste0('LSTM MAE: ', round(mean(abs(X - X_hybrid)), 3)))
print(paste0('HWI MAE: ', round(mean(abs(X - X_hwi)), 3)))








## EXAMPLE 2: Multiple gaps
## ---------------------

dev.off()
set.seed(10)
X = interpTools::simXt(N = 500, numTrend = 0)$Xt
X_gapped = interpTools::simulateGaps(list(X), p = 0.2, g = 25, K = 1)
X0 = X_gapped[[1]]$p0.2$g25[[1]]

plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)


## Defining parameters
window = 5
forecast = 1

## Generating training and testing data
data = data_generator(X0, window = window, forecast = forecast)

X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]

dim(X_train)
dim(Y_train)
dim(X_test_forward)
dim(X_test_backward)

## Fitting the model on training data
model = fit_model(X_train, Y_train, window, forecast, batch = 38, validation_split = 0.25)

## Creating rolling prediction window
rolling_prediction <- function(model, x, X_test, window, block){
  
  ## Defining helpful parameters
  preds = c()
  
  for (i in 1:dim(X_test)[1]){
    
    rolling_length = block[i]
    input = X_test[i,,]; dim(input) = c(1, window, 1)
    
    for (j in 1:rolling_length){
      
      ## Predicting on current input
      
      pred = model %>% predict(tf$constant(input), batch_size = 1, verbose = 0) %>% as.numeric()
      
      ## Appending results
      preds = c(preds, pred)
      
      ## Updating prediction series
      input = c(as.numeric(input), pred)[-1]
      dim(input) = c(1, window, 1)
    }
  }
  return(preds)
}

forward_preds = rolling_prediction(model, X0, X_test_forward, window, block = c(25, 25, 50))
backward_preds = rolling_prediction(model, X0, X_test_backward, window, block = c(50, 25, 25))

## Finalizing X_for and X_back
X_for = X; X_for[61:85] = forward_preds[1:25]; X_for[181:205] = forward_preds[26:50]; X_for[294:343] = forward_preds[51:100];
X_back = X; X_back[61:85] = rev(backward_preds[76:100]); X_back[181:205] = rev(backward_preds[51:75]); X_back[294:343] = rev(backward_preds[1:50]);

## Plotting the forward and backward predictions
plot_results <- function(){
  par(mfrow = c(2, 1), oma = c(4, 4, 4, 4), mar = c(2, 2, 2, 1), xpd = FALSE)
  
  plot(1:350, X[1:350], type = 'l', lty = 3, ylim = c(-30, 30),
       xlab = NA, ylab = NA); grid()
  lines(1:350, X0[1:350], type = 'l', lwd = 2)
  lines(61:85, X_for[61:85], type = 'l', col = 'dodgerblue', lwd = 1.5)
  lines(181:205, X_for[181:205], type = 'l', col = 'dodgerblue', lwd = 1.5)
  lines(294:343, X_for[294:343], type = 'l', col = 'dodgerblue', lwd = 1.5)
  legend('topright', legend = c('Forward Pass'), col = c('dodgerblue'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  plot(1:350, X[1:350], type = 'l', lty = 3, ylim = c(-30, 30),
       xlab = NA, ylab = NA); grid()
  lines(1:350, X0[1:350], type = 'l', lwd = 2)
  lines(61:85, X_back[61:85], type = 'l', col = 'red', lwd = 1.5)
  lines(181:205, X_back[181:205], type = 'l', col = 'red', lwd = 1.5)
  lines(294:343, X_back[294:343], type = 'l', col = 'red', lwd = 1.5)
  legend('topright', legend = c('Backward Pass'), col = c('red'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  title(main = 'LSTM Network Results', xlab = 'Time', ylab = 'X', 
        outer = TRUE, line = 0, cex.main = 2) 
}
plot_results()

## Computing prediction performance
X_hy = X0; X_hy[61:73] = X_for[61:73]; X_hy[74:85] = X_back[74:85]; X_hy[181:193] = X_for[181:193]; X_hy[194:205] = X_back[194:205]; X_hy[294:318] = X_for[294:318]; X_hy[319:343] = X_back[319:343]
X_hwi = interpTools::parInterpolate(X_gapped, methods = c('HWI'))[[1]]$HWI$p0.2$g25[[1]]

## Plotting forward/backward hybrid predictions vs. HWI imputation
plot_results2 <- function(){
  par(mfrow = c(2, 1), oma = c(4, 4, 4, 4), mar = c(2, 2, 2, 1), xpd = FALSE)
  
  plot(1:350, X[1:350], type = 'l', lty = 3, ylim = c(-30, 30),
       xlab = NA, ylab = NA); grid()
  lines(1:350, X0[1:350], type = 'l', lwd = 2)
  lines(61:85, X_hy[61:85], type = 'l', col = 'darkorange', lwd = 1.5)
  lines(181:205, X_hy[181:205], type = 'l', col = 'darkorange', lwd = 1.5)
  lines(294:343, X_hy[294:343], type = 'l', col = 'darkorange', lwd = 1.5)
  legend('topright', legend = c('LSTM Hybrid'), col = c('darkorange'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  plot(1:350, X[1:350], type = 'l', lty = 3, ylim = c(-30, 30),
       xlab = NA, ylab = NA); grid()
  lines(1:350, X0[1:350], type = 'l', lwd = 2)
  lines(61:85, X_hwi[61:85], type = 'l', col = 'forestgreen', lwd = 1.5)
  lines(181:205, X_hwi[181:205], type = 'l', col = 'forestgreen', lwd = 1.5)
  lines(294:343, X_hwi[294:343], type = 'l', col = 'forestgreen', lwd = 1.5)
  legend('topright', legend = c('HWI'), col = c('forestgreen'), 
         pch = 16, bty = 'n', cex = 1.2)
  
  title(main = 'LSTM Network Results', xlab = 'Time', ylab = 'X', 
        outer = TRUE, line = 0, cex.main = 2) 
}
plot_results2()

## Computing performance metrics
print(paste0('LSTM RMSE (forward): ', round(sqrt(mean((X - X_for)^2)), 3)))
print(paste0('LSTM RMSE (backward): ', round(sqrt(mean((X - X_back)^2)), 3)))
print(paste0('LSTM RMSE (hybrid): ', round(sqrt(mean((X - X_hy)^2)), 3)))
print(paste0('HWI RMSE: ', round(sqrt(mean((X - X_hwi)^2)), 3)))

print(paste0('LSTM MAE: ', round(mean(abs(X - X_for)), 3)))
print(paste0('LSTM MAE: ', round(mean(abs(X - X_back)), 3)))
print(paste0('LSTM MAE: ', round(mean(abs(X - X_hy)), 3)))
print(paste0('HWI MAE: ', round(mean(abs(X - X_hwi)), 3)))







## Something is off with the data generator for more than one-step-ahead forecasting
## EXAMPLE 3: Multi-step Forecast
## ---------------------

dev.off()
set.seed(10)
X = interpTools::simXt(N = 1000, numTrend = 0)$Xt
X_gapped = interpTools::simulateGaps(list(X), p = 0.05, g = 10, K = 1)
X0 = X_gapped[[1]]$p0.05$g10[[1]]

plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)


## Defining parameters
window = 25
forecast = 10

## Generating training and testing data
data = data_generator(X0, window = window, forecast = forecast)

X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]

dim(X_train)
dim(Y_train)
dim(X_test_forward)
dim(X_test_backward)





