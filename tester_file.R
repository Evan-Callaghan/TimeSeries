## TESTER


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)
library(keras)
library(tensorflow)
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

methods = c('KAF', 'EWMA', 'LI', 'LOCF', 'HWI', 'NNI')
N = c(500, 1000, 1500)
P = c(0.1, 0.2, 0.3)
G = c(5, 10, 20)
max_iter = 10
M = length(methods) * length(N) * length(P) * length(G) * max_iter
results = c()

for (n in N){
  for (p in P){
    for (g in G){
      for (iter in 1:max_iter){
        
        ## Simulating the time series
        x = interpTools::simXt(N = n, numTrend = 0)$Xt
        
        ## Scaling to 0-1
        x = (x - min(x)) / (max(x) - min(x))
        
        ## Imposing gap structure
        x0 = interpTools::simulateGaps(list(x), p = p, g = g, K = 1)
        
        ## Saving an alternate version of the gappy time series
        x0_alt = eval(parse(text = paste0('x0[[1]]$p', p, '$g', g, '[[1]]')))
        
        for (method in methods){
          
          print(paste0('N: ', n, ' P: ', p, ' G: ', g, ' Method: ', method, ' Iteration: ', iter))
          
          if (method == 'NNI'){
            ## Performing imputation with Neural Network Imputer (NNI)
            x_imp = main(x0_alt, max_iter = 10, n_series = 50, p = p, g = g, K = 4)
            
            ## Computing NNI performance
            performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
            
            ## Extracting performance metrics
            metrics = c()
            for (i in 1:length(performance)){
              metrics = append(metrics, array(performance)[[i]])
            }
            
            ## Appending results
            results = append(results, c(n, p, g, iter, method, metrics))
          }
          
          else{
            ## Performing imputation for all other specified methods
            x_imp = eval(parse(text = paste0('interpTools::parInterpolate(x0, methods = c(method))[[1]]$', 
                                             method, '$p', p, '$g', g, '[[1]]')))
            
            ## Computing performance
            performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
            
            ## Extracting performance metrics
            metrics = c()
            for (i in 1:length(performance)){
              metrics = append(metrics, array(performance)[[i]])
            }
            
            ## Appending results
            results = append(results, c(n, p, g, iter, method, metrics)) 
          }
        }
      }
    }
  }
}

# Experiment Notes:
# Neural Network Imputer (NNI) is using the 'noise' method of training data simulation with 
# random = TRUE to generate 200 time series (n_series * K). In the simulation process, the 
# modulus parameter comes from the Uniform(0.95, 1.05) and the argument parameter comes from 
# the Uniform(-pi/4, pi/4). 

## Cleaning the results 
results_df = as.data.frame(matrix(results, nrow = M, byrow = TRUE))
colnames(results_df) = c('N', 'P', 'G', 'K', 'Method', 'r', 'r_2', 'AD', 'MBE', 'ME', 'MAE', 
                         'MRE', 'MARE', 'MAPE', 'SSE', 'MSE', 'RMS', 'NMSE', 'RE', 'RMSE', 
                         'NRMSD', 'RMSS', 'MdAPE')
results_df[,c(-5)] = lapply(results_df[,c(-5)], as.numeric)


## Saving results file
write.csv(results_df, 'sim_results.csv', row.names = FALSE)

View(results_df)



grouped_results = results_df %>% dplyr::group_by(Method, N, P, G) %>%
  dplyr::summarize(mean_mae = mean(MAE), mean_rmse = mean(RMSE)) %>%
  dplyr::arrange(N, P, G, Method)

View(grouped_results)


results_df$MAE[1:10]
results_df$MAPE[1:10]







## UPDATES TO TESTING FILE BELOW ##

methods = c('HWI', 'EWMA', 'LI')
N = c(500)
P = c(0.1)
G = c(10, 25)
max_iter = 2
M = length(methods) * length(N) * length(P) * length(G) * max_iter
results = c()
results_NNI = c()

for (n in N){
  for (p in P){
    for (g in G){
      for (iter in 1:max_iter){
        
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
          x0_condition = (0 %in% x[missing_idx])
        }
        
        ## IMPUTATION ##
        
        for (method in methods){
          
          ## Performing imputation for all other specified methods
          x_imp = eval(parse(text = paste0('interpTools::parInterpolate(x0, methods = c(method))[[1]]$', 
                                           method, '$p', p, '$g', g, '[[1]]')))
          ## Computing performance
          performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
          
          ## Appending results
          results = append(results, c(n, p, g, iter, method, performance$MAE, performance$RMSE, performance$MAPE)) 
        }
        
        ## NEURAL NETWORK IMPUTATION ##
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 20, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 20, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 60, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 60, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 100, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 100, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ---------------------------- (Flipping 'K')
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 10, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 10, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 30, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 30, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 50, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 50, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ---------------------------- (Flipping 'max_iter')
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 20, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 20, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 60, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 60, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 100, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 100, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 10, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 10, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 30, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 30, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 50, 
                     random = TRUE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 50, TRUE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ---------------------------- (Flipping 'random')
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 20, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 20, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 60, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 60, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 100, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 100, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 10, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 10, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 30, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 30, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 50, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 50, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 20, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 20, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 60, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 60, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 100, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 100, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 10, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 10, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 30, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 30, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 50, 
                     random = FALSE, 
                     method = 'noise', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 50, FALSE, 'noise', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ---------------------------- (Flipping 'method')
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 20, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 20, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 60, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 60, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 100, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 100, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 10, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 10, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 30, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 30, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 50, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 50, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 20, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 20, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 60, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 60, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 100, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 100, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 10, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 10, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 30, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 30, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 50, 
                     random = TRUE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 50, TRUE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 20, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 20, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 60, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 60, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 5,
                     n_series = 100, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 5, 100, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 10, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 10, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 30, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 30, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 10, 
                     K = 10,
                     n_series = 50, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(10, 10, 50, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 20, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 20, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 60, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 60, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 5,
                     n_series = 100, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 5, 100, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 10, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 10, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 30, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 30, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
        
        ## Performing imputation with Neural Network Imputer (NNI)
        x_imp = main(x0 = x0_alt, p = p, g = g, 
                     max_iter = 25, 
                     K = 10,
                     n_series = 50, 
                     random = FALSE, 
                     method = 'all', 
                     Mod_a = 0.95, Mod_b = 1.05, 
                     Arg_a = -pi/6, Arg_b = pi/6)
        
        param_set = c(25, 10, 50, FALSE, 'all', 0.05, pi/6)
        
        ## Computing NNI performance
        performance = interpTools::eval_performance(x = x, X = x_imp, gappyx = x0_alt)
        
        ## Appending results
        results_NNI = append(results_NNI, c(n, p, g, iter, 'NNI', param_set, performance$MAE, performance$RMSE, performance$MAPE))
        
        ## ----------------------------
      }
    }
  }
}

## Cleaning the results 
results_df = as.data.frame(matrix(results, nrow = M, byrow = TRUE))
results_NNI_df = as.data.frame(matrix(results_NNI, nrow = , byrow = TRUE))

colnames(results_df) = c('N', 'P', 'G', 'Iteration', 'Method', 'MAE', 'RMSE', 'MAPE')
colnames(results_NNI_df) = c('N', 'P', 'G', 'Iteration', 'Method', 'max_iter', 'K', 'n_series', 'random', 'method', 
                             'Modulus', 'Argument', 'MAE', 'RMSE', 'MAPE')

results_df[,c(-5)] = lapply(results_df[,c(-5)], as.numeric)
results_NNI_df[,c(-5, -9, -10)] = lapply(results_NNI_df[,c(-5, -9, -10)], as.numeric)






