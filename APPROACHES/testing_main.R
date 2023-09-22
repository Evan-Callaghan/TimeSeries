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

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 5
METHODS = c('HWI', 'LI', 'LOCF')




## 1. Sunspots Data
sunspots = clean_ts(read.csv('Data/monthly-sunspots.csv')$Sunspots)
plot_ts(sunspots)

sunspots_sim = simulation_main(sunspots, P, G, K, METHODS)

sunspots_sim_x0 = sunspots_sim[[1]]
sunspots_sim_xI = sunspots_sim[[2]]
sunspots_sim_performance = sunspots_sim[[3]]
sunspots_sim_aggregation = sunspots_sim[[4]]
sunspots_sim_cleaned = simulation_cleaner(sunspots_sim_x0, sunspots_sim_xI, P, G, K, METHODS)
write.csv(sunspots_sim_cleaned, 'APPROACHES/SIMULATIONS/sunspots_sim.csv', row.names = FALSE)

sunspots_sim_plot = simulation_plot(sunspots_sim_aggregation, criteria = 'RMSE', agg = 'mean', 
                                    title = 'Sunspots Data Imputation:', levels = c('HWI', 'LI', 'LOCF'))
sunspots_sim_plot



sunspots_sim_data = data.frame(X = sunspots, 
                               x0 = as.numeric(sunspots_sim_cleaned[[1]][80,-c(1,2,3)]), 
                               HWI = as.numeric(sunspots_sim_cleaned[[2]][80, -c(1,2,3,4)]), 
                               LI = as.numeric(sunspots_sim_cleaned[[2]][160, -c(1,2,3,4)]), 
                               LOCF = as.numeric(sunspots_sim_cleaned[[2]][240, -c(1,2,3,4)]))
plot_series(sunspots_sim_data, title = 'Simulation Results: P=0.4, G=50')





## 2. Stock Price Data












plot_series <- function(data, title){
  plt = ggplot(data) +
    geom_line(aes(x = index(data), y = LOCF, color = 'LOCF')) +
    geom_line(aes(x = index(data), y = LI, color = 'LI')) +
    geom_line(aes(x = index(data), y = HWI, color = 'HWI')) +
    geom_line(aes(x = index(data), y = x0, color = 'X0'), linewidth = 1.01, alpha = 0.8) +
    labs(title = paste0(title), x = "Index", y = "Value") +
    scale_color_manual(name = "Legend", values = c("LOCF" = "red", "LI" = "dodgerblue", 'HWI' = 'green', 'X0' = 'black')) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0, face = 'bold', size = 18), 
          axis.text = element_text(color = 'black', size = 8), 
          axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'grey', linewidth = 0.5, linetype = 'dotted'))
  
  return(plt)
}









































