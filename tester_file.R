## TESTER


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)
library(keras)
library(tensorflow)
library(spsUtil)
options(warn = -1)

## Importing other functions
## -----------------------

source('estimate.R')
source('simulate.R')
source('impute.R')
source('initialize.R')
source('main.R')

## Imputation Framework:
## -----------------------

methods = c('HWI', 'EWMA', 'LI'); networks = 48

N = c(1000)
P = c(0.1, 0.2, 0.3)
G = c(10, 20, 30)
max_iter = 5

M = length(methods) * length(N) * length(P) * length(G) * max_iter
M_NNI = networks * length(N) * length(P) * length(G) * max_iter
results = c(); results_NNI = c()

for (n in N){
  for (p in P){
    for (g in G){
      for (iter in 1:max_iter){
        
        ## House-keeping
        print(paste0('N: ', n, '   P: ', p, '   G: ', g, '   Iteration: ', iter))
        set.seed(sample(x = 1:1e6, size = 1))
        
        ## Simulating the time series
        x = interpTools::simXt(N = n, numTrend = 0)$Xt
        
        ## Scaling to 0-1
        x = (x - min(x)) / (max(x) - min(x))
        
        x0_condition = TRUE
        while (x0_condition){
          
          ## Imposing gap structure
          x0 = interpTools::simulateGaps(list(x), p = p, g = g, K = 1)
          
          ## Saving an alternate version of the gapped time series
          x0_alt = eval(parse(text = paste0('x0[[1]]$p', p, '$g', g, '[[1]]')))
          
          ## Finding index of missing values
          missing_idx = which(is.na(x0_alt))
          
          ## Updating x0_condition
          x0_condition = (0 %in% x[missing_idx]) | (1 %in% missing_idx) | (length(missing_idx) %in% missing_idx)
        }
        
        ## IMPUTATION ##
        
        for (method in methods){
          
          ## Performing imputation for all other specified methods
          x_imp = quiet(eval(parse(text = paste0('interpTools::parInterpolate(x0, methods = c(method))[[1]]$', 
                                                 method, '$p', p, '$g', g, '[[1]]'))))
          ## Computing performance
          performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
          
          ## Appending results
          results = append(results, c(n, p, g, iter, method, performance$MAE, performance$RMSE, performance$MAPE)) 
        }
        
        ## NEURAL NETWORK IMPUTATION ##
        
        #' 'run_simulation' Arguments: x0, p, g, max_iter, K, n_series, random, method, Mod, Arg
        
        ## Default setup
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 20, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 60, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 100, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 10, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 30, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 50, TRUE, 'noise', 0.05, pi/6)))
        
        ## Flipping on 'max_iter'
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 20, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 60, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 100, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 10, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 30, TRUE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 50, TRUE, 'noise', 0.05, pi/6)))
        
        ## Flipping on 'random'
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 20, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 60, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 100, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 10, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 30, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 50, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 20, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 60, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 100, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 10, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 30, FALSE, 'noise', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 50, FALSE, 'noise', 0.05, pi/6)))
        
        ## Flipping on 'method'
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 20, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 60, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 100, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 10, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 30, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 50, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 20, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 60, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 100, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 10, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 30, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 50, TRUE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 20, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 60, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 5, 100, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 10, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 30, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 10, 10, 50, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 20, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 60, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 5, 100, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 10, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 30, FALSE, 'all', 0.05, pi/6)))
        results_NNI = append(results_NNI, c(n, p, g, iter, run_simulation(x0_alt, p, g, 25, 10, 50, FALSE, 'all', 0.05, pi/6)))
      }
    }
  }
}

## Cleaning the results 
results_df = as.data.frame(matrix(results, nrow = M, byrow = TRUE))
results_NNI_df = as.data.frame(matrix(results_NNI, nrow = M_NNI, byrow = TRUE))

colnames(results_df) = c('N', 'P', 'G', 'Iteration', 'Method', 'MAE', 'RMSE', 'MAPE')
colnames(results_NNI_df) = c('N', 'P', 'G', 'Iteration', 'Method', 'max_iter', 'K', 'n_series', 'random', 'method', 
                             'Modulus', 'Argument', 'MAE', 'RMSE', 'MAPE')

results_df[,c(-5)] = lapply(results_df[,c(-5)], as.numeric)
results_NNI_df[,c(-5, -9, -10)] = lapply(results_NNI_df[,c(-5, -9, -10)], as.numeric)


## Saving results file
write.csv(results_df, 'sim_results.csv', row.names = FALSE)
write.csv(results_NNI_df, 'sim_results_NNI.csv', row.names = FALSE)
