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
source('performance.R')


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
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
FRAMEWORK <- function(X, P, G, K, METHODS){
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K); print('Imposed Gaps')
  
  ## Impute
  xI = my_parInterpolate(x0, METHODS)
  
  ## Evaluate
  performance = my_performance(OriginalData = X, IntData = xI, GappyData = x0)
  
  ## Aggregate
  aggregation = interpTools::aggregate_pf(performance)
  
  ## Return
  return(list(x0, xI, performance, aggregation))
}

# X = interpTools::simXt(N = 1000, mu = 0)$Xt
# results = FRAMEWORK(X, P = c(0.1), G = c(10, 25), K = 3, METHODS = c('LI', 'HWI', 'NNI'))
# results
# my_new_multiHeatmap(results, P = c(0.1), G = c(10, 25), crit = 'RMSE', f = 'mean', 
#                     METHODS = c('LI', 'HWI', 'NNI'))


#' my_parInterpolate
#' 
#' Function which acts as a wrapper to the interpTools interpolation process and also adds the ability to use the Neural 
#' Network Imputer (NNI). 
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
my_parInterpolate <- function(x0, METHODS){
  
  ## If the list contains NNI...
  if ('NNI' %in% METHODS){
    
    ## Removing NNI from methods list
    METHODS = METHODS[METHODS != 'NNI']
    
    ## Calling interpTools with remaining methods
    xI_all = interpTools::parInterpolate(x0, methods = METHODS); print('Other Interpolation')
    
    ## Performing NNI imputation
    xI_NNI = my_parInterpolate_NNI(x0) ; print('NNI Interpolation')
    
    ## Joining the two results
    xI = list(c(xI_all[[1]], xI_NNI[[1]]))
  }
  
  else {
    
    ## Otherwise, just perform imputation with interpTools
    xI = interpTools::parInterpolate(x0, methods = METHODS)
  }
  
  ## Returning the imputed series
  return(xI)
}



#' my_parInterpolate_NNI
#' 
#' Function which acts as a wrapper to the NNI interpolation process. Takes an incomplete time series as input and uses NNI
#' to produce the interpolated series.
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#'
my_parInterpolate_NNI <- function(x0){
  
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
  function_call = paste0("run_simulation(x, 5, 500, 'all', 0.05, pi/6, 1)")
  
  ## Performing imputation
  int_series[[M]] = lapply(x0[[D]], function(x){
    lapply(x, function(x){
      lapply(x, function(x){
        eval(parse(text = function_call))})}
    )})
  
  ## Applying the function name 
  names(int_series) = c('NNI')
  
  ## Saving the imputed series
  int_data[[D]] = int_series
  
  ## Returning the imputed series
  return(int_data)
}


#' my_new_multiHeatmap
#' 
#' Function which takes an aggregations object as input and produces a multi heat map plot comparing 
#' performance across all methods. The user can specify the specific metric and aggregation function
#' to use (i.e., mean, median, IQR, etc.).
#' @param agg {aggregate_pf}; Aggregation object containing simulation results
#' @param P {list}; List of missing proportions to consider for the visualization
#' @param G {list}; List of gap width structures to consider for the visualization
#' @param crit {string}; Metric to consider from the aggregation object
#' @param f {string}; Aggregation type to be considered
#' @param title {string}; Any additional text to be added on to the end of the default title
#' @param colors {list}; List of plot colors to be considered for the visualization
#'
my_new_multiHeatmap <- function(agg, P, G, METHODS, crit = 'MAE', f = 'median', title = '', 
                                colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")){
  
  ## Initializing a data-frame 
  data = data.frame()
  
  ## Defining full color palette
  col = colorRampPalette(colors = colors)(100)
  
  ## Adding data from all aggregations
  for (p in P){
    for (g in G){
      for (method in METHODS){
        temp = eval(parse(text = paste0('as.data.frame(agg$D1$p', p, '$g', g, '$', method, ')')))
        temp$metric = rownames(temp); rownames(temp) = NULL
        data = rbind(data, temp)
      }
    }
  }
  ## Filtering the data-frame
  data = eval(parse(text = paste0('data %>% dplyr::select(method, prop_missing, gap_width, metric, ', f, ')'))) %>% 
    dplyr::filter(metric == crit) %>%
    dplyr::rename('P' = 'prop_missing', 'G' = 'gap_width')
  
  ## Creating plot
  theme_update(plot.title = element_text(hjust = 0.5))
  ggplot(data, aes(as.factor(P), as.factor(G), fill = data[,5])) +
    geom_tile(color = "gray95", lwd = 0.5, linetype = 1) +
    facet_grid(~ method) +
    labs(title = paste0('Simulation Results ', title), 
         x = "Missing Proportion (P)", 
         y = "Gap Width (G)", 
         fill = data[1,4]) +
    scale_fill_gradientn(colours = col, values = c(0,1)) + 
    guides(fill = guide_colourbar(label = TRUE, ticks = TRUE)) +
    geom_text(aes(label = round(data[,5], 3)), color = "black", size = 3) +
    theme_minimal() + 
    theme(plot.title = element_text(hjust = 0.5, size = 18)) +
    theme(strip.text = element_text(face = 'bold', size = 10))
}

# plot = my_new_multiHeatmap(results, P = c(0.1, 0.15, 0.2, 0.25, 0.3), G = c(10, 25, 50), 
#                     METHODS = c('LI', 'EWMA', 'LOCF'), f = 'mean', crit = 'MAE')
# plot






