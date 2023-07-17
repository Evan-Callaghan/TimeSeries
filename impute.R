########################################
## Neural Network Time Series Imputer ##
########################################

## Imputer: All code related to constructing and fitting the neural network imputer


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)

## Notation Notes
## -----------------------

#' @param x0 {list}; List containing the original incomplete time series ("x naught")

## Defining all functions
## -----------------------

#' impute
#' 
#' Function to construct, compile, and fit a neural network model on the simulated training data
#' and then predict on the original time series. Completes the fifth step of the designed algorithm.
#' @param x0 {list}; List containing the original incomplete time series ("x naught")
#' @param inputs {matrix}; Matrix object containing the input training data
#' @param targets {matrix}; Matrix object containing the target training data
#' 
imputer <- function(x0, inputs, targets, model){
  
  ## Defining useful parameters
  N = dim(inputs)[2]; EPOCHS = 25; BATCH_SIZE = 32
  
  x0 = as.array(matrix(ifelse(is.na(x0), 0, x0), nrow = 1, byrow = TRUE)) ## Formatting original series
  
  inputs = tf$constant(inputs) ## Creating input tensors
  targets = tf$constant(targets) ## Creating target tensors
  x0 = tf$constant(x0) ## Creating prediction tensor
  
  ## Compiling the model
  model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')
  
  ## Fitting the model to the training data
  model %>% fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE, shuffle = TRUE, 
                validation_split = 0.2, verbose = 0)
  
  preds = model %>% predict(x0, verbose = 0) ## Predicting on the original series
  return(preds)
}


get_model <- function(Architecture, N){
  
  autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
    layer_masking(mask_value = 0, input_shape = c(N), name = 'mask') %>%
    layer_dense(units = 64, activation = 'relu', name = 'encoder1') %>%
    layer_dense(units = 32, activation = 'relu', name = 'encoder2') %>%
    layer_dense(units = 64, activation = 'relu', name = 'decoder1') %>%
    layer_dense(units = N, activation = 'linear', name = 'decoder2')
  return(autoencoder)
  
  if (Architecture == 1){
    autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
      #layer_masking(mask_value = -1, input_shape = c(N), name = 'mask') %>%
      layer_masking(mask_value = 0, input_shape = c(N), name = 'mask') %>%
      layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
      layer_dense(units = N, activation = 'linear', name = 'decoder')
    return(autoencoder)
  }
  else if(Architecture == 2){
    autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
      layer_masking(mask_value = 0, input_shape = c(N), name = 'mask') %>%
      layer_dense(units = 64, activation = 'relu', name = 'encoder1') %>%
      layer_dense(units = 32, activation = 'relu', name = 'encoder2') %>%
      layer_dense(units = 64, activation = 'relu', name = 'decoder1') %>%
      layer_dense(units = N, activation = 'linear', name = 'decoder2')
    return(autoencoder)
  }
  else if(Architecture == 3){
    autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
      layer_masking(mask_value = -1, input_shape = c(N,1), name = 'mask') %>%
      layer_lstm(units = 64, name = 'LSTM') %>%
      layer_dense(units = 32, activation = 'relu', name = 'encoder') %>% 
      layer_dense(units = N, activation = 'sigmoid', name = 'decoder')
    return(autoencoder)
  }
  else if(Architecture == 4){
    autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
      layer_masking(mask_value = -1, input_shape = c(N), name = 'mask') %>%
      layer_dropout(0.2, name = 'dropout')
    layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
      layer_dense(units = N, activation = 'sigmoid', name = 'decoder')
    return(autoencoder)
  }
}




