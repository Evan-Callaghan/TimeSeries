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
imputer <- function(x0, inputs, targets){
  
  ## Defining useful parameters
  N = dim(inputs)[2]
  EPOCHS = 10; BATCH_SIZE = 32
  
  x0 = as.array(matrix(ifelse(is.na(x0), -1, x0), nrow = 1, byrow = TRUE)) ## Formatting original series
  
  inputs = tf$constant(inputs) ## Creating input tensors
  targets = tf$constant(targets) ## Creating target tensors
  x0 = tf$constant(x0) ## Creating prediction tensor
  
  autoencoder = keras_model_sequential(name = 'Autoencoder') %>% ## Constructing the model
    layer_masking(mask_value = -1, input_shape = c(N), name = 'mask') %>%
    layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
    layer_dense(units = N, activation = 'sigmoid', name = 'decoder') 
  
  autoencoder %>% compile( ## Compiling the model
    optimizer = 'adam', loss = 'binary_crossentropy')

  autoencoder %>% fit(inputs, targets, epochs = EPOCHS, ## Fitting the model to the training data
                      batch_size = BATCH_SIZE, shuffle = TRUE, validation_split = 0.2, verbose = 1)
  
  preds = autoencoder %>% predict(x0, verbose = 0) ## Predicting on the original series
  return(preds)
}