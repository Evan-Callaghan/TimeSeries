##########################################
## Baseline Simulation Helper Functions ##
##########################################


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
simulation_main <- function(X, P, G, K, METHODS, numCores){
  
  ## Setting common seed
  set.seed(42)
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K)
  
  ## Impute
  xI = interpTools::parInterpolate(x0, methods = METHODS, numCores = numCores)
  
  ## Evaluate
  performance = simulation_performance(X, x0, xI)
  
  ## Save
  results = simulation_saver(performance, P, G, K, METHODS)
  
  ## Return
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
          
          x0_temp = x0[[1]][[p]][[g]][[k]]
          xI_temp = xI[[1]][[m]][[p]][[g]][[k]]
          
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
  X = X[index]
  xI = xI[index]
  
  # Defining helpful parameters
  N = length(X)
  return <- list()
  
  # Computing performance metrics:
  
  # Coefficient of Correlation, r
  numerator = sum((xI - mean(xI))*(X - mean(X)))
  denominator = sqrt(sum((xI - mean(xI))^2)) * sqrt(sum((X - mean(X))^2))
  return$pearson_r = numerator / denominator
  
  # r^2
  return$r_squared = return$pearson_r^2  
  
  # Absolute Differences
  return$AD = sum(abs(xI - X))
  
  # Mean Bias Error 
  return$MBE = sum(xI - X) / N
  
  # Mean Error 
  return$ME = sum(X - xI) / N
  
  # Mean Absolute Error 
  return$MAE = abs(sum(X - xI)) / N
  
  # Mean Relative Error 
  if (length(which(X == 0)) == 0) {
    return$MRE = sum((X - xI) / X)  
  } else {
    return$MRE = NA
  }
  
  # Mean Absolute Relative Error ##### Lepot
  if (length(which(X == 0)) == 0) {
    return$MARE = 1/N*sum(abs((X - xI) / X))
  } else {
    return$MARE = NA 
  }
  
  # Mean Absolute Percentage Error 
  return$MAPE = 100 * return$MARE
  
  # Sum of Squared Errors
  return$SSE = sum((xI - X)^2)
  
  # Mean Square Error 
  return$MSE = 1 / N * return$SSE
  
  # Root Mean Squares, or Root Mean Square Errors of Prediction 
  if (length(which(X == 0)) == 0) {
    return$RMS = sqrt(1 / N * sum(((xI - X)/X)^2))
  } else {
    return$RMS = NA 
  }
  
  # Mean Squares Error (different from MSE, referred to as NMSE)
  return$NMSE = sum((X - xI)^2) / sum((X - mean(X))^2)
  
  # Reduction of Error, also known as Nash-Sutcliffe coefficient 
  return$RE = 1 - return$NMSE
  
  # Root Mean Square Error, also known as Root Mean Square Deviations
  return$RMSE = sqrt(return$MSE)
  
  # Normalized Root Mean Square Deviations 
  return$NRMSD = 100 * (return$RMSE / (max(X) - min(X)))
  
  # Root Mean Square Standardized Error 
  if (sd(X) != 0) {
    return$RMSS = sqrt(1 / N * sum(((xI - X)/sd(X) )^2))  
  } else {
    return$RMSS = NA 
  }
  
  # Median Absolute Percentage Error
  if (length(which(X == 0)) == 0) {
    return$MdAPE = median(abs((X - xI) / X))*100  
  } else {
    return$MdAPE = NA
  }
  
  # Log-Cosh Loss
  return$LCL <- sum(log(cosh(xI - X))) / N
  
  # Returning the final list of all performance metrics
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
  colnames(performance_summary) = c('Method', 'P', 'G', 'K', 'pearson_r', 'r_squared', 'AD', 'MBE', 'ME', 'MAE', 'MRE', 'MARE',
                                    'MAPE', 'SSE', 'MSE', 'RMS', 'NMSE', 'RE', 'RMSE', 'NRMSD', 'RMSS', 'MdAPE', 'LCL')
  return(performance_summary)
}




