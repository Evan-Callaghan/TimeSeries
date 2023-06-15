########################################
## Neural Network Time Series Imputer ##
########################################

## MAIN METHOD: 


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)

## Importing other functions
## -----------------------

source('estimate.R')
source('simulate.R')
source('impute.R')
source('initialize.R')

## Defining all functions
## -----------------------

#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param n_series {integer}; Number of new time series to construct
#' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#'
main <- function(x0, max_iter, n_series, p, g, K){
  
  # Defining matrix to store results
  results = matrix(NA, ncol = length(x0), nrow = max_iter)
  
  ## Step 1: Linear imputation
  xV = initialize(x0)
  
  for (i in 1:max_iter){
    
    ## Step 2/3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xV, n_series, p, g, K, random = TRUE, method = 'noise')
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    pred = imputer(x0, inputs, targets)
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), pred, x0); results[i,] = xV
    #print(paste0('Iteration ', i))
  }
  return(colMeans(results))
}
