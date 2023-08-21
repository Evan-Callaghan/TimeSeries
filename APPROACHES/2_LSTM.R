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
#' @param x0 {list}; List object containing the original incomplete time series
#'
main <- function(x0, max_iter, train_size, method = 'all', Mod, Arg, Architecture){
  
  ## Defining useful variables
  N = length(x0)
  
  ## Defining matrix to store results
  results = matrix(NA, ncol = N, nrow = max_iter)
  return(colMeans(results))
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















set.seed(10)
X = interpTools::simXt(N = 500, numFreq = 2, b = c(100, 200))$Xt
X0 = X; X0[200:249] = NA

plot(X, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)





block = blocking(X0, window = 5, forecast = 1)
block



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
  
  ## Generating training data
  for (idx in train_indices){
    
    N = block[idx,]$Gap
    start_idx = block[idx,]$Start
    end_idx = block[idx,]$End
    
    x_temp = x[start_idx:end_idx]
    
    for (i in 1:(N-window)){
      row = x_temp[i:(i+window-1)]
      label = x_temp[(i+window)]
      X_train = c(X_train, row)
      Y_train = c(Y_train, label)
    }
  }
  X_train = array(matrix(X_train, nrow = M, byrow = TRUE), dim = c(M, window))
  Y_train = array(matrix(Y_train, nrow = M, byrow = TRUE), dim = c(M, 1))
  
  
  ## Generating testing data
  for (idx in test_indices){
    
    start_idx = block[idx,]$Start - window
    end_idx = block[idx,]$Start - forecast
    
    X_test = c(X_test, x[start_idx:end_idx])
  }
  X_test = array(matrix(X_test, nrow = n_test_indices, byrow = TRUE), dim = c(n_test_indices, window, 1))
  
  ## Return statement
  return(list(X_train, Y_train, X_test))
}



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
  dim(Y_train) = c(M, 1)
  
  return(list(X_train, Y_train, X_test_for, X_test_back))
}

data = data_generator(X0, window = 5, forecast = 1)


X_train = data[[1]]
Y_train = data[[2]]
X_test_forward = data[[3]]
X_test_backward = data[[4]]


X0
rev(X0)






X_train[430:440,,]
Y_train[430:440,]
X_test

dim(X_train)
dim(Y_train)
dim(X_test_forward)
dim(X_test_backward)

## How do we properly sort the X_test_forward and backward matrices? Makes a difference
## when we have more than one missing region.







