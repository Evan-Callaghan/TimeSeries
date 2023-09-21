##################
## Testing File ##
##################


## Importing libraries
## -----------------------

library(dplyr)
library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)


## Configuring set-up (not always necessary)
## -----------------------

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}


## Importing functions
## -----------------------

source('APPROACHES/1_Autoencoder_updated.R')


## Simulations
## -----------------------

P = c(0.1, 0.2, 0.3)
G = c(5, 10, 25)
K = 10
METHODS = c('LOCF', 'LI', 'HWI')


## 1. Sunspots Data

























set.seed(42)
X = interpTools::simXt(N = 500)$Xt
X_0_1 = clean_ts(X)

plot1 = plot_ts(X)
plot2 = plot_ts(X_0_1)
grid.arrange(plot1, plot2, nrow = 2)


results = simulation_main(X, P, G, K, METHODS)

x0 = results[[1]]
xI = results[[2]]
performance = results[[3]]
aggregation = results[[4]]
cleaned_exp = simulation_cleaner(x0, xI, P, G, K, METHODS)


results_0_1 = simulation_main(X_0_1, P, G, K, METHODS)

x0_0_1 = results_0_1[[1]]
xI_0_1 = results_0_1[[2]]
performance_0_1 = results_0_1[[3]]
aggregation_0_1 = results_0_1[[4]]
cleaned_exp_0_1 = simulation_cleaner(x0_0_1, xI_0_1, P, G, K, METHODS)


plot1 = simulation_plot(aggregation, criteria = 'RMSE', agg = 'mean', 
                        title = 'Original Data', levels = c('LOCF', 'LI', 'HWI'))
plot2 = simulation_plot(aggregation_0_1, criteria = 'RMSE', agg = 'mean', 
                        title = 'Scaled Data', levels = c('LOCF', 'LI', 'HWI'))
grid.arrange(plot1, plot2, nrow = 2)



no_scale = data.frame(X = X,
                      x0 = as.numeric(cleaned_exp[[1]][42,-seq(1, 3)]), 
                      locf = as.numeric(cleaned_exp[[2]][42,-seq(1, 4)]), 
                      li = as.numeric(cleaned_exp[[2]][132,-seq(1, 4)]), 
                      hwi = as.numeric(cleaned_exp[[2]][222,-seq(1, 4)]))
scaled = data.frame(X = X_0_1,
                    x0 = as.numeric(cleaned_exp_0_1[[1]][42,-seq(1, 3)]), 
                    locf = as.numeric(cleaned_exp_0_1[[2]][42,-seq(1, 4)]), 
                    li = as.numeric(cleaned_exp_0_1[[2]][132,-seq(1, 4)]), 
                    hwi = as.numeric(cleaned_exp_0_1[[2]][222,-seq(1, 4)]))

plot_series <- function(data){
  plt = ggplot(data) +
    geom_line(aes(x = index(data), y = locf, color = 'LOCF')) +
    geom_line(aes(x = index(data), y = li, color = 'LI')) +
    geom_line(aes(x = index(data), y = hwi, color = 'HWI')) +
    geom_line(aes(x = index(data), y = x0, color = 'X0'), linewidth = 1.01, alpha = 0.8) +
    labs(x = "Index", y = "Value") +
    scale_color_manual(name = "Legend", values = c("LOCF" = "red", "LI" = "dodgerblue", 'HWI' = 'green', 'X0' = 'black')) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0, face = 'bold', size = 18), 
          axis.text = element_text(color = 'black', size = 8), 
          axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'grey', linewidth = 0.5, linetype = 'dotted'))
  
  return(plt)
}

plot1 = plot_series(no_scale)
plot2 = plot_series(scaled)
grid.arrange(plot1, plot2, nrow = 2)
































## Simulations:
## ---------------------


P = c(0.1, 0.2, 0.3)
G = c(5, 10)
K = 3
METHODS = c('HWI', 'LI', 'NNI')

set.seed(42)
X = interpTools::simXt(N = 1000)$Xt
plot_ts(X)

results = simulation_main(X, P, G, K, METHODS)

x0 = results[[1]]
xI = results[[2]]
performance = results[[3]]
aggregation = results[[4]]

simulation_plot(aggregation, criteria = 'RMSE', agg = 'mean', 
                title = 'Simulation Results:', levels = c('HWI', 'LI', 'NNI'))









## Time Series:

## Create function:
## Scales the time series
## Return a plot
## Maybe remove titles from these.
## Uniform dimensions when exporting.
## Want them to be the same size in latex.

ts = read.csv('monthly-sunspots.csv')
ts = ts$Sunspots

X_t = data.frame(index = seq(1, length(ts)), value = ts)

ggplot(data = X_t, aes(x = index, y = value)) +
  geom_line(color = "#204176") + 
  geom_point(color = '#204176', size = 0.3) +
  labs(title = paste0('Time Series: Monthly Sunspot Data'), x = "Index", y = "Value") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0, face = 'bold', size = 18), 
        axis.text = element_text(color = 'black', size = 8), 
        axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
        axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
        panel.grid = element_line(color = 'grey', linewidth = 0.5, linetype = 'dotted'))


plot_ts(ts, 'test')






