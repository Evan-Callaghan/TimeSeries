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
K = 10
METHODS = c('HWI', 'LI', 'LOCF')




## Simulation 1: Sunspots Data
## -----------------------

## Reading and visualizing the data
sunspots = read.csv('Data/monthly-sunspots.csv')$Sunspots
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








## Generated time series with high SNR value
set.seed(5000)
X = interpTools::simXt(N = 1000, numTrend = 0, snr = 10000)$Xt
plot_ts(X)


## Generated time series with low SNR value
set.seed(111)
X = interpTools::simXt(N = 1000, numTrend = 0, snr = 0.01)$Xt
plot_ts(X)

## Sunspots time series (https://www.kaggle.com/datasets/hugoherrera11/monthly-sunspots) 
sunspots = read.csv('Data/monthly-sunspots.csv')$Sunspots
plot_ts(sunspots)


## Room temperature time series (https://www.kaggle.com/datasets/vitthalmadane/ts-temp-1)
temperature = read.csv('Data/MLTempDataset.csv')
head(temperature)
plot_ts(temperature$DAYTON_MW)


## Price of gold time series (https://www.kaggle.com/datasets/arashnic/learn-time-series-forecasting-from-gold-price)
gold = read.csv('Data/gold_price_data.csv')
head(gold)
plot_ts(gold$Value)








t = 1:1000
X_mod = 10*cos(2*pi*(0.001*t + (1/20000*t^2 - 0.05*t)))

set.seed(11)
W_t = interpTools::simWt(N = 1000, var = 20)
plot_ts(X_mod + W_t$value)








sim_Tt_mod <- function(N = 1000, numFreq, P){
  
  ## Defining helpful variables
  Tt_list <- list()
  t = 0:(N-1)
  fourierFreq = 2*pi/N
  P = P + 1
  text = ''
  
  for (i in 1:numFreq){
    
    mu = round(rnorm(1, mean = 0, sd = N/200), 3)
    f = round(runif(1, fourierFreq, pi), 3)
    a = round(rnorm(P, mean = 0, sd = N/200), 3)
    
    text = paste0(text, "(", mu, ")*cos(2*", round(pi, 3), "*(", f, "*t + ", sep = "", collapse = "")
    
    for (p in 1:P){
      text = paste0(text, a[p] / (p), "*t^", p, " + ", sep = "", collapse = "")
    }
    
    text = paste0(text, '0)) + ')
  }
  
  text = substr(text, start = 1, stop = nchar(text) - 3)
  
  print(text)
  
  ## Returning final list object
  Tt_list$fn <- paste(text, collapse="")  
  Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
  return(Tt_list)
}


Tt = sim_Tt_mod(N = 1000, numFreq = 1, P = 0)
plot_ts(Tt$value)





## Generating a time series with a frequency modulated signal
t = 1:1000
sequence = 50 * cos(2*pi*(0.03)*t + (2*pi*(1/1000*t^2 - t)))
plot_ts(sequence)












N = 25
t <- 0:(N-1)
fourierFreq <- 2*pi/N

mu <- 2
f <- 0.5
P <- 3
a = c(0.1, 0.12, 0.13)
temp = (a[1] / 2 * t^(2)) + (a[2] / 3 * t^(3)) + (a[3] / 4 * t^(4))
X = mu * cos(2*pi*(f*t*temp))
plot(X, type = 'l', lwd = 2); grid()

a <- rnorm(P, 0, 1)

temp = rep(0, N)
for (p in 1:P){
  temp = temp + (a[p] / (p+1) * t^(p+1))
}

to_return = mu * cos(2*pi*(f*t*temp))

paste("(",b,")*sin(", w,"*t)+", sep = "", collapse = "")



N = 100
t = 0:(N-1)
numFreq = 2
fourierFreq = 2*pi/N
P = 2
text = ''
for (i in 1:numFreq){
  mu = rpois(1, 4)
  
  mu = rnorm(numFreq, mean = 0, sd = N/200)
  f = runif(numFreq, fourierFreq, pi)
  
  f = round(runif(1, 0.5, 1.5), 2)
  a = rpois(P+1, 4)
  print(a)
  
  text = paste0(text, "(", mu, ")*cos(2*", pi, "*(", f, "*t + ", sep = "", collapse = "")
  
  for (p in 1:(P+1)){
    
    text = paste0(text, a[p] / (p), "*t^", p, " + ", sep = "", collapse = "")
  }
  
  text = paste0(text, '0)) + ')
}
text = substr(text, start = 1, stop = nchar(text) - 3)
print(text)

text_fn <- paste(text, collapse = "")  
text_value <- eval(parse(text = paste(text, collapse = "")))
plot(text_value, type = 'l')



