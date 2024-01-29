#################################
## Simulation Helper Functions ##
#################################


## Defining all functions
## -----------------------


#' simulation_main
#' 
#' Function which facilitates the testing of all methods implemented in the interpTools package.
#' @param X {list}; List object containing a complete time series (should be scaled to (0,1))
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports all methods from interpTools)
#'
simulation <- function(X, P, G, K, METHODS, numCores){
  
  # Setting common seed
  set.seed(23)
  
  # Impose
  X0 = interpTools::simulateGaps(list(X), P, G, K)
  
  # Impute
  XI = interpTools::parInterpolate(X0, methods = METHODS, numCores = numCores)
  
  # Evaluate
  performance = simulation_performance(X, X0, XI)
  
  # Save
  results = simulation_saver(performance, P, G, K, METHODS)
  
  # Return
  return(results)
}


#' simulation_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to compute imputation performance with a variety of loss functions / performance metrics.
#' @param X {list}; List object containing the original complete time series
#' @param x0 {list}; List object containing the original incomplete time series
#' @param xI {list}; List object containing the interpolated time series
#'
simulation_performance <- function(X, x0, xI){
  
  # Defining helpful parameters
  M = length(xI[[1]])
  P = length(xI[[1]][[1]])
  G = length(xI[[1]][[1]][[1]])
  K = length(xI[[1]][[1]][[1]][[1]])
  
  # Initializing nested list object
  pf = lapply(pf <- vector(mode = 'list', M), function(x) 
    lapply(pf <- vector(mode = 'list', P), function(x) 
      lapply(pf <- vector(mode = 'list', G), function(x)
        x <- vector(mode = 'list', K))))
  
  # Creating vector names
  prop_vec_names = numeric(P)
  gap_vec_names = numeric(G)
  method_names = numeric(M)
  
  prop_vec = as.numeric(gsub("p", "", names(xI[[1]][[1]])))
  gap_vec = as.numeric(gsub("g", "", names(xI[[1]][[1]][[1]])))
  method_names = names(xI[[1]])
  
  # Evaluating the performance criteria for each sample in every (m, p, g) specification
  for(m in 1:M){
    for(p in 1:P){
      prop_vec_names[p] = c(paste("p", prop_vec[p], sep = "")) # vector of names
      for(g in 1:G){
        gap_vec_names[g] = c(paste("g", gap_vec[g], sep = "")) # vector of names
        for(k in 1:K) { 
          
          x0_temp = as.numeric(x0[[1]][[p]][[g]][[k]])
          xI_temp = as.numeric(xI[[1]][[m]][[p]][[g]][[k]])
          
          pf[[m]][[p]][[g]][[k]] = unlist(simulation_performance_helper(X, x0_temp, xI_temp))
        }
        names(pf[[m]][[p]]) = gap_vec_names
      }
      names(pf[[m]]) = prop_vec_names
    }
    names(pf) = method_names
  }
  
  # Doing a final cleaning of the nested list object
  pf = lapply(pf, function(x) 
    lapply(x, function(x) 
      lapply(x, function(x)
        lapply(x, function(x){
          logic = unlist(lapply(x, FUN = function(x) !is.null(x)))
          x = x[logic]}))))
  class(pf) = 'pf'
  
  # Returning all performance metrics
  return(pf) 
}


#' simulation_performance_helper
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to aggregate imputer performance over several iterations and combinations of P, G, and K and return
#' a neatly laid out performance report.
#' @param X {list}; List object containing the original complete time series
#' @param x0 {list}; List object containing the original incomplete time series
#' @param xI {list}; List object containing the interpolated time series
#'
simulation_performance_helper <- function(X, x0, xI) {
  
  # Identifying indices with NA values
  index = which(is.na(x0))
  
  # Only considering indices with NA values
  X = X[index]; xI = xI[index]
  
  # Defining helpful parameters
  N = length(X)
  return <- list()
  
  # Computing performance metrics:
  
  # Mean Absolute Error 
  return$MAE = sum(abs(xI - X)) / N
  
  # Root Mean Square Error
  return$RMSE = sqrt(sum((xI - X)^2) / N)
  
  # Log-Cosh Loss
  return$LCL = sum(log(cosh(xI - X))) / N
  
  # Returning the final list of performance metrics
  return(return)
}


#' simulation_saver
#' 
#' Function to store all simulation results in a consistent format. Takes the simulation performance as input
#' and sorts through the nested lists to organize imputation performance in a data-frame object. The returned 
#' data-frame can be easily exported to save simulation results.
#' @param performance {list}; Nested list object containing performance evaluation from the the set of simulations
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports all methods from interpTools)
#'
simulation_saver <- function(performance, P, G, K, METHODS){
  
  # Initializing data-frame to store results
  performance_summary = data.frame()
  
  # Looping through all simulation combinations
  for (method in METHODS){
    for (p in P){
      for (g in G){
        for (k in 1:K){
          
          ## Appending the selected performance metrics
          metrics = paste0("performance$", method, "$p", p, "$g", g, "[[", k, "]]")
          performance_summary = rbind(performance_summary, c(method, p, g, k, as.numeric(eval(parse(text = metrics)))))
        }
      }
    }
  }
  # Cleaning the final data-frame
  colnames(performance_summary) = c('Method', 'P', 'G', 'K', 'MAE', 'SSE', 'MSE','RMSE','LCL','MAE2','RMSE2','LCL2')
  return(performance_summary)
}









P = c(0.1)
G = c(5, 10)
K = 2
METHODS = c('NN', 'LI', 'NCS', 'FMM', 'HCS', 'SI', 'KAF', 'KKSF', 'LOCF', 'NOCB', 'SMA', 
            'LWMA', 'EWMA', 'RMEA', 'RMED', 'RMOD', 'RRND', 'HWI')
numCores = 16


# 4. High SNR Data

# Reading time series data-frames
high_snr = read.csv('Simulations/Preliminary/Data/high_snr_data.csv')

# Performing interpolation
high_snr_results = simulation(high_snr$data, P, G, K, METHODS, numCores)



View(high_snr_results)

