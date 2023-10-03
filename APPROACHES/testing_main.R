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


## Configuring GPU set-up
## -----------------------

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}


## Importing functions
## -----------------------

source('APPROACHES/1_Autoencoder_updated.R')


## Simulation Parameters
## -----------------------

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 50
METHODS = c('HWI', 'LI', 'LOCF')



## Simulation 1: Sunspots Data
## -----------------------

## Reading and visualizing the data
sunspots = clean_ts(read.csv('Data/monthly-sunspots.csv')$Sunspots)
plot_ts(sunspots)

## Running the imputation simulation
sunspots_sim = simulation_main(sunspots, P, G, K, METHODS)

## Saving simulation results
sunspots_sim_x0 = sunspots_sim[[1]]
sunspots_sim_xI = sunspots_sim[[2]]
sunspots_sim_performance = sunspots_sim[[3]]
sunspots_sim_aggregation = sunspots_sim[[4]]

## Saving simulation performance and exporting as a csv file
sunspots_saved = simulation_saver(sunspots_sim_performance, P, G, K, METHODS)
write.csv(sunspots_saved, 'APPROACHES/SIMULATIONS/sunspots_performance.csv', row.names = FALSE)

## Creating aggregation evaluation plot
sunspots_sim_plot = simulation_plot(sunspots_sim_aggregation, criteria = 'RMSE', agg = 'mean', 
                                    title = 'Sunspots Data Imputation:', levels = c('HWI', 'LI', 'LOCF'))
sunspots_sim_plot


View(sunspots_saved)





## Generated time series with high SNR value
set.seed(5000)
X = interpTools::simXt(N = 1000, numTrend = 0, snr = 10000)$Xt
plot_ts(clean_ts(X), title = 'High SNR Generated Data')


## Generated time series with low SNR value
set.seed(111)
X = interpTools::simXt(N = 1000, numTrend = 0, snr = 0.01)$Xt
plot_ts(clean_ts(X), title = 'Low SNR Generated Data')

## Sunspots time series (https://www.kaggle.com/datasets/hugoherrera11/monthly-sunspots) 
sunspots = read.csv('Data/monthly-sunspots.csv')$Sunspots
plot_ts(clean_ts(sunspots), title = 'Monthly Sunspots Data')


## Daily Price of Apple stock time series (Yahoo Finance, 5YR, Daily)
apple = read.csv('Data/AAPL.csv')
head(apple)
plot_ts(clean_ts(apple$Close), title = 'Apple Stock Price Data')


## Frequency modulated time series
t = 1:1000
X_mod = 10*cos(2*pi*(0.001*t + (1/20000*t^2 - 0.05*t)))
set.seed(11)
W_t = interpTools::simWt(N = 1000, var = 20)
X = X_mod + W_t$value
plot_ts(clean_ts(X), title = 'Frequency Modulated Generated Data')








