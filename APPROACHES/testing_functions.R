####################################
## Functions for Mass Simulations ##
####################################


## Importing libraries
## -----------------------

library(dplyr)
library(tsinterp)
library(interpTools)


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
simulation_main <- function(X, P, G, K, METHODS){
  
  ## Setting common seed
  set.seed(42)
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K)
  
  ## Impute
  xI = simulation_impute(x0, METHODS)
  
  ## Evaluate
  performance = simulation_performance(X = X, xI = xI, x0 = x0)
  
  ## Save
  results = simulation_saver(performance, P, G, K, METHODS)
  
  ## Return
  return(results)
}


#' simulation_impute
#' 
#' Function which acts as a wrapper to the interpTools interpolation process.
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#' @param METHODS {list}; List of method to consider for imputation (supports all methods from interpTools)
#'
simulation_impute <- function(x0, METHODS){
  
  ## Performing imputation with interpTools
  xI = interpTools::parInterpolate(x0, methods = METHODS)
  
  ## Returning the set of imputed series
  return(xI)
}


#' simulation_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to compute imputation performance with a variety of loss functions / performance metrics.
#' @param X {list}; List object containing the original complete time series
#' @param xI {list}; List object containing the interpolated time series
#' @param x0 {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
simulation_performance <- function(X, xI, x0, custom = NULL){
  
  D <- length(xI)
  M <- length(xI[[1]])
  P <- length(xI[[1]][[1]])
  G <- length(xI[[1]][[1]][[1]])
  K <- length(xI[[1]][[1]][[1]][[1]])
  
  # Initializing nested list object
  pf <- lapply(pf <- vector(mode = 'list', D),function(x)
    lapply(pf <- vector(mode = 'list', M),function(x) 
      lapply(pf <- vector(mode = 'list', P),function(x) 
        lapply(pf <- vector(mode = 'list', G),function(x)
          x<-vector(mode='list', K)))))
  
  prop_vec_names <- numeric(P)
  gap_vec_names <- numeric(G)
  method_names <- numeric(M)
  data_names <- numeric(D)
  
  prop_vec <- as.numeric(gsub("p","",names(xI[[1]][[1]])))
  gap_vec <- as.numeric(gsub("g","",names(xI[[1]][[1]][[1]])))
  method_names <- names(xI[[1]])
  
  if(is.null(names(xI))){
    data_names <- paste0("D", 1:D)
  }
  else{
    data_names <- names(xI)
  }
  
  # Evaluate the performance criteria for each sample in each (d,m,p,g) specification
  for(d in 1:D){
    for(m in 1:M){
      for(p in 1:P){
        prop_vec_names[p] <- c(paste("p", prop_vec[p],sep="")) # vector of names
        for(g in 1:G){
          gap_vec_names[g] <- c(paste("g", gap_vec[g],sep="")) # vector of names
          for(k in 1:K) { 
            #pf[[d]][[m]][[p]][[g]][[k]] <- unlist(eval_performance(x = OriginalData[[d]], X = IntData[[d]][[m]][[p]][[g]][[k]], gappyx = GappyData[[d]][[p]][[g]][[k]], custom = custom))
            
            ## Editing here -----------
            ## Removing the "[[d]]" from OriginalData
            ## Changing to "my_eval_performance" after changing the function names from their originals
            pf[[d]][[m]][[p]][[g]][[k]] <- unlist(simulation_eval_performance(x = X, X = xI[[d]][[m]][[p]][[g]][[k]], gappyx = x0[[d]][[p]][[g]][[k]], custom = custom))
          }
          names(pf[[d]][[m]][[p]]) <- gap_vec_names
        }
        names(pf[[d]][[m]]) <- prop_vec_names
      }
      names(pf[[d]]) <- method_names
    }
    names(pf) <- data_names
  }
  
  pf <- lapply(pf, function(x) 
    lapply(x, function(x) 
      lapply(x, function(x)
        lapply(x, function(x){
          logic <- unlist(lapply(x,FUN = function(x) !is.null(x)))
          x <-x[logic]
        }))))
  
  class(pf) <- "pf"
  return(pf) 
}


#' simulation_eval_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to aggregate imputer performance over several iterations and combinations of P, G, and K and return
#' a neatly laid out performance report.
#' @param x {list}; List object containing the original complete time series
#' @param X {list}; List object containing the interpolated time series
#' @param GappyData {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
simulation_eval_performance <- function(x, X, gappyx, custom = NULL) {
  
  # x = original , X = interpolated 
  
  if(sum(is.na(gappyx)) == 0) stop(paste0("Gappy data in 'gappyx' does not contain NAs. Please impose gaps and try again."))
  if(sum(x - gappyx, na.rm = TRUE) != 0) stop(paste0("Gappy data in 'gappyx' is not representative of 'x' (original data). The two vectors are non-conforming."))
  #if(sum(X[which(!is.na(gappyx))] - x[which(!is.na(gappyx))]) != 0) stop(paste0("Non-interpolated points in 'X' do not match those of the original data in 'x'.  The two vectors are non-conforming."))
  
  if(!is.null(X)){
    stopifnot((is.numeric(x) | is.null(x)),
              (is.numeric(X) | is.null(X)),
              (is.numeric(gappyx) | is.null(gappyx)),
              length(x) == length(X),
              length(gappyx) == length(x), 
              length(gappyx) == length(X))
    
    # identify which values were interpolated
    index <- which(is.na(gappyx))
    
    # only consider values which have been replaced
    X <- X[index]
    x <- x[index]
    
    n <- length(x)
    
    return <- list()
    
    # Coefficent of Correlation, r
    numerator <- sum((X - mean(X))*(x - mean(x)))
    denominator <- sqrt(sum((X - mean(X))^2)) * sqrt(sum((x - mean(x))^2))
    return$pearson_r <- numerator / denominator
    
    # r^2
    return$r_squared <- return$pearson_r^2  
    
    # Absolute Differences
    return$AD <- sum(abs(X - x))
    
    # Mean Bias Error 
    return$MBE <- sum(X - x) / n
    
    # Mean Error 
    return$ME <- sum(x - X) / n
    
    # Mean Absolute Error 
    return$MAE <- abs(sum(x - X)) / length(x)
    
    # Mean Relative Error 
    if (length(which(x == 0)) == 0) {
      return$MRE <- sum((x - X) / x)  
    } else {
      return$MRE <- NA
    }
    
    # Mean Absolute Relative Error ##### Lepot
    if (length(which(x == 0)) == 0) {
      return$MARE <- 1/length(x)*sum(abs((x - X) / x))
    } else {
      return$MARE <- NA 
    }
    
    # Mean Absolute Percentage Error 
    return$MAPE <- 100 * return$MARE
    
    # Sum of Squared Errors
    return$SSE <- sum((X - x)^2)
    
    # Mean Square Error 
    return$MSE <- 1 / n * return$SSE
    
    # Root Mean Squares, or Root Mean Square Errors of Prediction 
    if (length(which(x == 0)) == 0) {
      return$RMS <- sqrt(1 / n * sum(((X - x)/x)^2))
    } else {
      return$RMS <- NA 
    }
    
    # Mean Squares Error (different from MSE, referred to as NMSE)
    return$NMSE <- sum((x - X)^2) / sum((x - mean(x))^2)
    
    # Reduction of Error, also known as Nash-Sutcliffe coefficient 
    return$RE <- 1 - return$NMSE
    
    # Root Mean Square Error, also known as Root Mean Square Deviations
    return$RMSE <- sqrt(return$MSE)
    
    # Normalized Root Mean Square Deviations 
    return$NRMSD <- 100 * (return$RMSE / (max(x) - min(x)))
    
    # Root Mean Square Standardized Error 
    if (sd(x) != 0) {
      return$RMSS <- sqrt(1 / n * sum(( (X-x)/sd(x) )^2))  
    } else {
      return$RMSS <- NA 
    }
    
    # Median Absolute Percentage Error
    if (length(which(x == 0)) == 0) {
      return$MdAPE <- median(abs((x - X) / x))*100  
    } else {
      return$MdAPE <- NA
    }
    
    ######### ADDITIONS:
    ## Log-Cosh Loss
    return$LCL <- sum(log(cosh(X - x))) / n
    
    
    # Custom functions
    
    if(!is.null(custom)){
      
      ####################
      ### LOGICAL CHECKS
      ####################
      
      n_custom <- length(custom)
      
      if(n_custom == 1){
        is_fn <- !inherits(try(match.fun(custom), silent = TRUE), "try-error") # FALSE if not a function
      }
      else if(n_custom > 1){
        is_fn <- logical(n_custom)
        for(k in 1:n_custom){
          is_fn[k] <- !inherits(try(match.fun(custom[k]), silent = TRUE), "try-error") # FALSE if not a function
        }
      }
      
      if(!all(is_fn)){
        not <- which(!is_fn)
        stop(c("Custom function(s): ", paste0(custom[not], sep = " ") ,", are not of class 'function'."))
      }
      
      # Check that the output of the function is a single value
      
      check_single <- function(fn){
        
        x <- rnorm(10)
        X <- rnorm(10)
        
        val <- match.fun(fn)(x = x, X = X)
        
        return(all(length(val) == 1, is.numeric(val)))
      }
      
      logic <- logical(n_custom)
      
      for(k in 1:n_custom){
        logic[k] <- check_single(custom[k])
      }
      
      if(!all(logic)){
        stop(c("Custom function(s): ", paste0(custom[!logic], sep = " "), ", do not return a single numeric value."))
      }
      
      # Computing custom metric values
      
      return_call <- character(n_custom)
      
      for(k in 1:n_custom){
        
        return_call[k] <- paste0("return$",custom[k]," <- match.fun(",custom[k],")(x = x, X = X)")
        
        eval(parse(text = return_call[k]))
      }
    }
    return(return)
  }
  else if(is.null(X)){
    return(NULL)
  }
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
  
  ## Initializing data-frame to store results
  performance_summary = data.frame()
  
  ## Looping through all simulation combinations
  for (method in METHODS){
    for (p in P){
      for (g in G){
        for (k in 1:K){
          
          ## Appending the selected performance metrics
          metrics = paste0("performance$D1$", method, "$p", p, "$g", g, "[[", k, "]]")
          performance_summary = rbind(performance_summary, c(method, p, g, k, as.numeric(eval(parse(text = metrics)))))
        }
      }
    }
  }
  ## Cleaning the final data-frame
  colnames(performance_summary) = c('Method', 'P', 'G', 'K', 'pearson_r', 'r_squared', 'AD', 'MBE', 'ME', 'MAE', 'MRE', 'MARE',
                                    'MAPE', 'SSE', 'MSE', 'RMS', 'NMSE', 'RE', 'RMSE', 'NRMSD', 'RMSS', 'MdAPE', 'LCL')
  return(performance_summary)
}


#' clean_ts
#' 
#' Function to normalize the input time series to the range (0,1). 
#' @param x {list}; List object containing the original complete time series
#'
clean_ts <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}









