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
  for (idx in test_indices){
    start_idx = block[idx,]$Start - window
    end_idx = block[idx,]$Start - 1
    X_test = c(X_test, x[start_idx:end_idx])
  }
  X_test = array(matrix(X_test, nrow = n_test_indices, byrow = TRUE), dim = c(n_test_indices, window, 1))
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
  X_test_for = tf$constant(X_test_for)
  X_test_back = tf$constant(X_test_back)
  
  return(list(X_train, Y_train, X_test_for, X_test_back))
}







fit_model <- function(X_train, Y_train){
  ## Defining parameters
  BATCH = 20  ## Must be a factor of length(training_data)
  WINDOW = 5
  
  ## Building the model
  model = keras_model_sequential(name = 'Model') %>% 
    layer_lstm(64, stateful = FALSE, batch_input_shape = c(BATCH, WINDOW, 1)) %>%
    layer_dense(8, activation = 'relu') %>%
    layer_dense(1, activation = 'linear')
  
  model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                    loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')
  
  model %>% fit(X_train, Y_train, verbose = 0, epochs = 50, batch_size = BATCH)
  
  fitted_weights = model %>% get_weights()
  
  single_item_model = keras_model_sequential(name = 'Model') %>% 
    layer_lstm(64, stateful = FALSE, batch_input_shape = c(1, WINDOW, 1)) %>%
    layer_dense(8, activation = 'relu') %>%
    layer_dense(1, activation = 'linear')
  
  single_item_model %>% set_weights(fitted_weights)
  
  single_item_model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                                loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')
  return(single_item_model)
}






X0
rev(X0)



## EDITS:

## How do we properly sort the X_test_forward and backward matrices? Makes a difference
## when we have more than one missing region.

## Remember to use tf$constant() to create tensor objects




set.seed(10)
X = interpTools::simTt(N = 500, numFreq = 2, b = c(100, 200))$value
X0 = X; X0[c(200:224, 375:399)] = NA

plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)


data = data_generator(X0, window = 10, forecast = 1)

X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]



X_test_backward




for (i in 5:1){
  ## Perform rolling window
  ## reverse preds
  
}




## EXAMPLE:
## ---------------------
set.seed(10)
X = interpTools::simTt(N = 500, numFreq = 2, b = c(100, 200))$value
X0 = X; X0[200:249] = NA

plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)


## Generating training and testing data
data = data_generator(X0, window = 10, forecast = 1)

X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]

dim(X_train)
dim(Y_train)
dim(X_test_forward)
dim(X_test_backward)



X_train[185:189,,]
Y_train[185:189,]

X_test_forward


















## Fitting the model on training data
model = fit_model(X_train, Y_train)

## Creating rolling prediction window (forward pass)
n_missing = sum(is.na(X0))
interps = c()
for (i in 1:n_missing){
  
  ## Predicting on current input
  test_preds = model %>% predict(X_test_forward, batch_size = 1, verbose = 0)
  
  ## Appending results
  interps = c(interps, test_preds)
  
  ## Updating 
  X_test_forward = c(X_test_forward, test_preds)[-1]
  dim(X_test_forward) = c(1, 5, 1)
}

interps


plot(150:249, X[150:249], type = 'l', col = 'red'); grid()
lines(150:249, X0[150:249], type = 'l', lwd = 2)
lines(200:249, interps, type = 'l', col = 'dodgerblue')






## Visualizing predictions 
train_preds = model %>% predict(X_train, batch_size = BATCH, verbose = 0)
valid_preds = model %>% predict(X_test_forward, verbose = 0)

plot(1:880, Y_train, type = 'l', lwd = 2, xlim = c(1, 880), main = 'Sunspots Data with LSTM Model', 
     ylab = 'Sunspots', xlab = 'Month'); grid()
lines(1201:1485, Y_valid, type = 'l', lwd = 2)
lines(1:1200, train_preds, type = 'l', col = 'red')
lines(1201:1485, valid_preds, type = 'l', col = 'dodgerblue')
legend(legend = c('Training', 'Validation'), lty = 1, lwd = 4, cex = 0.8,
       col = c('red', 'dodgerblue'), 'topleft', title = 'LSTM Predictions')








data = interpTools::simXt(N = 1000)$Xt
window = 20
forecast = 5
X = c(); Y = c()
M = length(data) - window
P = forecast - 1
for (i in 1:M){
  row = data[i:(i+window-1)]
  label = data[(i+window):(i+window+P)]
  X = c(X, row)
  Y = c(Y, label)
}
X = array(matrix(X, nrow = M, byrow = TRUE), dim = c(M, window, 1))
Y = array(matrix(Y, nrow = M, byrow = TRUE), dim = c(M, forecast))

dim(X)
dim(Y)

X[1:5,,]
Y[1:5,]










