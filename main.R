########################################
## Neural Network Time Series Imputer ##
########################################

## MAIN METHOD: 


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)


## Defining all functions
## -----------------------

#' #' main
#' #' 
#' #' Function that works through the designed algorithm and calls functions in order.
#' #' @param x0 {list}; List object containing the original incomplete time series
#' #' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' #' @param n_series {integer}; Number of new time series to construct
#' #' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' #' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' #' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#' #' @param random {boolean}; Indicator of whether or not the gap placement is randomized
#' #' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' #' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' #' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' #'
#' main <- function(x0, max_iter, n_series, p, g, K, random = TRUE, method = 'noise', Mod, Arg, Architecture){
#'   
#'   # Defining matrix to store results
#'   results = matrix(NA, ncol = length(x0), nrow = max_iter)
#'   
#'   ## Step 1: Linear imputation
#'   xV = initialize(x0)
#'   
#'   for (i in 1:max_iter){
#'     
#'     ## Step 2/3/4: Simulating time series and imposing gap structure
#'     data = simulator(x0, xV, n_series, p, g, K, random, method, Mod, Arg)
#'     inputs = data[[1]]; targets = data[[2]]
#'     
#'     ## Step 5: Performing the imputation
#'     pred = imputer(x0, inputs, targets, Architecture)
#'     
#'     ## Step 6: Extracting the predicted values and updating imputed series
#'     xV = ifelse(is.na(x0), pred, x0); results[i,] = xV
#'     #print(paste0('Iteration ', i))
#'   }
#'   return(colMeans(results))
#' }
#' 
#' 
#' #' run_simulation
#' #' 
#' #' Function to create a more organized method for running consecutive simulations. Calls the 'main' function to go through 
#' #' all neural network imputer steps and then returns simulation results. Takes all the same parameters as 'main.'
#' #' @param x0 {list}; List object containing the original incomplete time series
#' #' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' #' @param n_series {integer}; Number of new time series to construct
#' #' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' #' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' #' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#' #' @param random {boolean}; Indicator of whether or not the gap placement is randomized
#' #' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' #' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' #' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' #'
#' run_simulation <- function(x0, p, g, max_iter, K, n_series, random, method, Mod, Arg, Architecture){
#'   
#'   print(sum(is.na(x0)) / length(x0))
#'   print(sum(is.na(x0)))
#'   return(x0)
#'   
#'   ## Performing imputation with Neural Network Imputer (NNI)
#'   x_imp = main(x0 = x0, p = p, g = g, 
#'                max_iter = max_iter, 
#'                K = K,
#'                n_series = n_series, 
#'                random = random, 
#'                method = method, 
#'                Mod = Mod,
#'                Arg = Arg, 
#'                Architecture = Architecture)
#'   
#'   return(x_imp)
#'   
#'   ## Computing NNI performance
#'   #performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
#'   
#'   ## Storing results
#'   #results = c('NNI', max_iter, K, n_series, random, method, Mod, Arg, performance$MAE, performance$RMSE, performance$MAPE)
#'   #results = c(performance$MAE, performance$RMSE, performance$MAPE)
#'   
#'   #return(results)
#' }




#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#'
main <- function(x0, max_iter, train_size, method = 'all', Mod, Arg, Architecture){
  
  ## Defining useful variables
  N = length(x0)
  
  ## Defining matrix to store results
  results = matrix(NA, ncol = N, nrow = max_iter)
  
  ## Building the neural network model
  model = get_model(Architecture, N)
  
  ## Step 1: Linear imputation
  xV = initialize(x0)
  
  for (i in 1:max_iter){
    
    ## Step 2/3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xV, train_size, method, Mod, Arg)
    inputs = data[[1]]; targets = data[[2]]; Mt = data[[3]]; mean = data[[4]]
    
    ## Step 5: Performing the imputation
    pred = imputer(x0 = x0 - Mt - mean, inputs, targets, model)
    
    ## Adjusting 'pred' to include trend and mean
    pred_adj = pred[1,] + Mt + mean 
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), pred_adj, x0); results[i,] = xV
    #print(paste0('Iteration ', i))
  }
  return(colMeans(results))
}


#' run_simulation
#' 
#' Function to create a more organized method for running consecutive simulations. Calls the 'main' function to go through 
#' all neural network imputer steps and then returns simulation results. Takes all the same parameters as 'main.'
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#'
run_simulation <- function(x0, max_iter, train_size, method, Mod, Arg, Architecture){
  
  ## Performing imputation with Neural Network Imputer (NNI)
  x_imp = main(x0 = x0,
               max_iter = max_iter, 
               train_size = train_size, 
               method = method, 
               Mod = Mod,
               Arg = Arg, 
               Architecture = Architecture)
  
  ## Returning imputed series
  return(x_imp)
}

