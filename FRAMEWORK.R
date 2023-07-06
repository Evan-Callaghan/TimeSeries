########################################
## Neural Network Time Series Imputer ##
########################################

## TESTING FRAMEWORK: 


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)
source('estimate.R')
source('simulate.R')
source('impute.R')
source('initialize.R')
source('main.R')


## Defining all functions
## -----------------------


#' FRAMEWORK
#' 
#' Function which facilitates the testing of the Neural Network Imputer (NNI) versus other methods implemented in the
#' interpTools package.
#' @param X {list}; List object containing a complete time series (should be scaled to (0,1))
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {string}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
FRAMEWORK <- function(X, P, G, K, METHODS){
  
  ## Saving length of experiment
  M = length(P) * length(G) * K * length(METHODS)
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), p, g, K)
  
  ## Impute
  xI = my_parInterpolate(x0, METHODS)
  
  ## Evaluate
  performance = my_performance(X, xI, x0)
  
  ## Aggregate
  aggregation = interpTools::aggregate_pf(performance)
  
  ## Return
  return(aggregation)
}
X = interpTools::simXt(N = 1000, mu = 0)$Xt
FRAMEWORK(X, P = c(0.1), G = c(5, 10), K = 10, METHODS = c('HWI', 'LI', 'EWMA', 'NNI'))

# interpTools::plotMetrics(agg, p = 0.2, g = 10, metric = 'RMSE')
# interpTools::plotMetrics(agg, p = 0.2, g = 10, metric = 'MAE')








####################
my_parInterpolate <- function(x0, METHODS){
  
  if ('NNI' %in% METHODS){
    METHODS = METHODS[METHODS != 'NNI'] ## Removing 'NNI' from methods list
    xI_all = interpTools::parInterpolate(x0, methods = METHODS) ## Calling interpTools with remaining methods
    xI_NNI = my_parInterpolate_NNI(x0) ## NNI imputation with custom function
    xI = list(c(xI_all[[1]], xI_NNI[[1]])) ## Joining the two results
    return(xI)
  }
  
  else {
    xI = interpTools::parInterpolate(x0, methods = METHODS)
    return(xI)
  }
}


my_parInterpolate_NNI <- function(x0){
  
  fun_names = c('NNI') ## Defining naming convention
  
  ## Defining helpful variables
  D = 1
  M = 1
  P = length(x0[[1]])
  G = length(x0[[1]][[1]])
  K = length(x0[[1]][[1]][[1]])
  numCores = detectCores()
  
  ## Initializing lists to store interpolated series
  int_series = lapply(int_series <- vector(mode = 'list', M), function(x)
    lapply(int_series <- vector(mode = 'list', P), function(x) 
      lapply(int_series <- vector(mode = 'list', G), function(x) 
        x <- vector(mode = 'list', K))))
  
  int_data = list()
  
  ## Setting up the function call
  function_call = paste0("run_simulation(x, 10, 300, 'noise', 0.05, pi/6, 1)")
  #function_call = paste0("imputeTS::na.locf(option = 'locf', x = ", "x", ")")
  print(function_call)
  
  ## Parallel computing over K 
  # int_series[[M]] = lapply(x0[[D]], function(x){
  #   lapply(x, function(x){
  #     mclapply(x, function(x){
  #       eval(parse(text = function_call))}, mc.cores = numCores)}
  #   )})
  int_series[[M]] = lapply(x0[[D]], function(x){
    lapply(x, function(x){
      lapply(x, function(x){
        eval(parse(text = function_call))})}
    )})
  
  ## Applying the function name 
  names(int_series) = c(fun_names)
  
  ## Saving the imputed series
  int_data[[D]] = int_series
  
  return(int_data)
}

####################

#my_parInterpolate_NNI(x0)

ret = my_parInterpolate(x0, METHODS = c('LI', 'EWMA', 'NNI'))

ret


perf = my_performance(Xt, ret, x0)

agg = interpTools::aggregate_pf(perf)
agg
