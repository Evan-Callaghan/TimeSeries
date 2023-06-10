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
    data = simulator(x0, xV, n_series, p, g, K, random = TRUE, method = 'all')
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    pred = imputer(x0, inputs, targets)
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), pred, x0); results[i,] = xV
    print(paste0('Iteration ', i))
  }
  return(colMeans(results))
}













set.seed(42)
x = simXt(N = 250, mu = 0, numTrend = 1, numFreq = 2)$Xt
x = (x - min(x)) / (max(x) - min(x))

x_0 = simulateGaps(list(x), p = 0.1, g = 10, K = 1)
x0 = x_0[[1]]$p0.1$g10[[1]]
plot(x, type = 'l', col = 'red'); grid()
lines(x0, type = 'l')

imp = main(x0, max_iter = 20, n_series = 100, p = 0.1, g = 10, K = 5)

plot(x, type = 'l', lwd = 2); grid()
lines(imp, col = 'green', type = 'l', lwd = 0.7)



interp = parInterpolate(x_0, methods = c('HWI'))
lines(interp[[1]]$HWI$p0.1$g10[[1]], type = 'l', lwd = 0.7, col = 'red')



data = simulator(x_0, x, n_series = 100, p = 0.1, g = 10, K = 5, random = TRUE, method = 'noise')
inputs = data[[1]]; targets = data[[2]]

imputed = imputer(x_0, inputs, targets)
lines(imputed[1,], type = 'l', col = 'red')







data = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'noise')
data2 = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'all')
data3 = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'separate')
lines(data[[2]][1,], type = 'l', col = 'darkorange')
lines(data[[2]][2,], type = 'l', col = 'darkorange')
lines(data2[[2]][1,], type = 'l', col = 'dodgerblue')
lines(data2[[2]][2,], type = 'l', col = 'dodgerblue')
lines(data3[[2]][1,], type = 'l', col = 'red')
lines(data3[[2]][2,], type = 'l', col = 'red')



mt = estimateMt(x, N = length(x), nw = 5, k = 8, pMax = 2)
lines(mt, type = 'l', col = 'red')
detrend = x - mt
plot(detrend, type = 'l')
mean(detrend)
plot(detrend - mean(detrend), type = 'l')

no_mea = detrend - mean(detrend)
mean(no_mea)
