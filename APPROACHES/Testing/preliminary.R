##########################################
## Autoencoder Preliminary Testing File ##
##########################################



## Importing libraries
## -----------------------

library(tensorflow)
library(keras)
library(interpTools)
# install_tensorflow(envname = "r-tensorflow")


## Configuring GPU set-up
## -----------------------

## NOTE: R Project set up to use virtual Python version
# gpus = tf$config$experimental$list_physical_devices('GPU')
# for (gpu in gpus){
#   tf$config$experimental$set_memory_growth(gpu, TRUE)
# }



## Importing functions
## -----------------------

source('Simulations/Preliminary/Code/autoencoder_helper.R')


## Generating Time Series Data-Frames
## -----------------------

generate_df <- function(data, P, G, K){
  
  # Defining helpful parameters
  N = length(data); M = length(P) * length(G) * K
  
  # Initializing
  names = c(); columns = c()
  
  # Scaling the input data
  data_scaled = (data - min(data)) / (max(data) - min(data))
  
  # Simulating gaps using interpTools
  set.seed(222)
  data0 = interpTools::simulateGaps(list(data_scaled), P, G, K)
  
  # Extracting data from interpTools nested lists
  for (p in P){
    for (g in G){
      for (k in 1:K){
        names = c(names, paste0('X', p, '_', g, '_', k))
        function_call = paste0('columns = c(columns, data0[[1]]$p', p, '$g', g, '[[', k, ']])')
        eval(parse(text = function_call))
      }
    }
  }
  # Formatting the data-frame with proper column names
  data0 = as.data.frame(matrix(columns, nrow = N, byrow = FALSE))
  colnames(data0) = names
  
  # Formatting the complete time series into a data-frame
  data_df = data.frame('data' = data_scaled)
  
  # Sanity Check:
  for (m in 1:M){
    print(paste0('P:', estimate_p(data0[,m]), '  G:', estimate_g(data0[,m])))
  }
  
  return(list(data_df, data0))
}


## Simulation Parameters
## -----------------------

MODELS = c(1)
TRAIN_SIZE = c(500, 1000, 2000)
BATCH_SIZE = c(25, 50, 100)

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 5



## Simulation 1: Sunspots Data
## -----------------------

# Reading sunspots time series data
sunspots = read.csv('Data/Cleaned/sunspots.csv')$sunspots

# Creating sunspots data-frames
sunspots_df = generate_df(sunspots, P, G, K)
data = sunspots_df[[1]]; data0 = sunspots_df[[2]]

# Running the imputation simulation
sunspots_sim = simulation(data, data0, MODELS, TRAIN_SIZE, BATCH_SIZE)

# Exporting simulation performance as a csv file
write.csv(sunspots_sim, 'Simulations/Preliminary/Results/Prelim_sunspots.csv', row.names = FALSE)

# Observed simulation time: ____
View(sunspots_sim)



## Simulation 2: Apple Stock Data
## -----------------------

# Reading apple time series data
apple = read.csv('Data/Cleaned/apple.csv')$apple

# Creating apple data-frames
apple_df = generate_df(apple, P, G, K)
data = apple_df[[1]]; data0 = apple_df[[2]]

# Running the imputation simulation
apple_sim = simulation(data, data0[,c(5,6)], MODELS, TRAIN_SIZE, BATCH_SIZE)

# Exporting simulation performance as a csv file
write.csv(apple_sim, 'Simulations/Preliminary/Results/Prelim_apple.csv', row.names = FALSE)

# Observed simulation time: ____
View(apple_sim)



## Simulation 3: Temperature Data
## -----------------------

# Creating custom models for temperature data
model1 = create_model(N = length(temperature), units = 64, connected_units = 32, 'relu')
model2 = create_model(N = length(temperature), units = 128, connected_units = 32, 'relu')
MODELS = c(model1, model2)

# Running the imputation simulation
tic('Temperature Simulation')
temperature_sim = simulation_main(temperature, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
toc()

# Exporting simulation performance as a csv file
write.csv(temperature_sim, 'Simulations/Preliminary/Results/Prelim_November2023_temperature.csv', row.names = FALSE)

# Observed simulation time: ____



## Simulation 4: High SNR Time Series
## -----------------------

# Creating custom models for high SNR data
model1 = create_model(N = length(high_snr), units = 64, connected_units = 32, 'relu')
model2 = create_model(N = length(high_snr), units = 128, connected_units = 32, 'relu')
model3 = create_model(N = length(high_snr), units = 256, connected_units = 32, 'relu')
MODELS = c(model1, model2, model3)

# Running the imputation simulation
tic('High SNR Simulation')
high_snr_sim = simulation_main(high_snr, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
toc()

# Exporting simulation performance as a csv file
write.csv(high_snr_sim, 'Simulations/Preliminary/Results/Prelim_November2023_high_snr.csv', row.names = FALSE)

# Observed simulation time: ____



## Simulation 5: Low SNR Time Series
## -----------------------

# Creating custom models for low SNR data
model1 = create_model(N = length(low_snr), units = 64, connected_units = 32, 'relu')
model2 = create_model(N = length(low_snr), units = 128, connected_units = 32, 'relu')
MODELS = c(model1, model2)

# Running the imputation simulation
tic('Low SNR Simulation')
low_snr_sim = simulation_main(low_snr, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
toc()

# Exporting simulation performance as a csv file
write.csv(low_snr_sim, 'Simulations/Preliminary/Results/Prelim_November2023_low_snr.csv', row.names = FALSE)

# Observed simulation time: ____



## Simulation 6: Frequency Modulated Time Series
## -----------------------

# Creating custom models for modulated data
model1 = create_model(N = length(modulated), units = 64, connected_units = 32, 'relu')
model2 = create_model(N = length(modulated), units = 128, connected_units = 32, 'relu')
MODELS = c(model1, model2)

## Running the imputation simulation
tic('Modulated Simulation')
modulated_sim = simulation_main(modulated, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
toc()

## Exporting simulation performance as a csv file
write.csv(modulated_sim, 'Simulations/Preliminary/Results/Prelim_November2023_modulated.csv', row.names = FALSE)

# Observed simulation time: ____










## Generating CSV files
## -----------------------

generate_df <- function(data, P, G, K){
  
  N = length(data)
  names = c(); columns = c()
  
  data0 = interpTools::simulateGaps(list(data), P, G, K)
  
  for (p in P){
    for (g in G){
      for (k in 1:K){
        names = c(names, paste0(p, '_', g, '_', k))
        function_call = paste0('columns = c(columns, data0[[1]]$p', p, '$g', g, '[[', k, ']])')
        eval(parse(text = function_call))
      }
    }
  }
  data0 = as.data.frame(matrix(columns, nrow = N, byrow = FALSE))
  colnames(data0) = names
  
  data_df = data.frame('data' = data)
  
  return(list(data_df, data0))
}

## -----------------------

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 5

sunspots = read.csv('Data/Cleaned/sunspots.csv')$sunspots
apple = read.csv('Data/Cleaned/apple.csv')$apple
temperature = read.csv('Data/Cleaned/toronto_temperature.csv')$temperature
high_snr = read.csv('Data/Cleaned/simulated.csv')$high_snr
low_snr = read.csv('Data/Cleaned/simulated.csv')$low_snr
modulated = read.csv('Data/Cleaned/simulated.csv')$modulated


# Sunspots:
sunspots_data = generate_df(sunspots, P, G, K)

write.csv(sunspots_data[[1]], 'Data/Exported/sunspots_data.csv', row.names = FALSE)
write.csv(sunspots_data[[2]], 'Data/Exported/sunspots_data0.csv', row.names = FALSE)


# Apple:
apple_data = generate_df(apple, P, G, K)

write.csv(apple_data[[1]], 'Data/Exported/apple_data.csv', row.names = FALSE)
write.csv(apple_data[[2]], 'Data/Exported/apple_data0.csv', row.names = FALSE)


# Temperature:
temperature_data = generate_df(temperature, P, G, K)

write.csv(temperature_data[[1]], 'Data/Exported/temperature_data.csv', row.names = FALSE)
write.csv(temperature_data[[2]], 'Data/Exported/temperature_data0.csv', row.names = FALSE)


# High SNR:
high_snr_data = generate_df(high_snr, P, G, K)

write.csv(high_snr_data[[1]], 'Data/Exported/high_snr_data.csv', row.names = FALSE)
write.csv(high_snr_data[[2]], 'Data/Exported/high_snr_data0.csv', row.names = FALSE)


# Low SNR:
low_snr_data = generate_df(low_snr, P, G, K)

write.csv(low_snr_data[[1]], 'Data/Exported/low_snr_data.csv', row.names = FALSE)
write.csv(low_snr_data[[2]], 'Data/Exported/low_snr_data0.csv', row.names = FALSE)


# Modulated:
modulated_data = generate_df(modulated, P, G, K)

write.csv(modulated_data[[1]], 'Data/Exported/modulated_data.csv', row.names = FALSE)
write.csv(modulated_data[[2]], 'Data/Exported/modulated_data0.csv', row.names = FALSE)








library(dplyr)

data = read.csv('Data/Prelim_Autoencoder_November2023_sunspots.csv')
dim(data)

data_grouped = data %>% dplyr::group_by(P, G, Model, Train.Size, Batch.Size) %>% 
  dplyr::summarise(MAE = mean(MAE), RMSE = mean(RMSE), LCL = mean(LCL))

View(data_grouped %>% dplyr::filter(P == 0.1, G == 5))
View(data_grouped %>% dplyr::filter(P == 0.2, G == 25))


library(gt)

P = 0.25

data_filtered = data_grouped %>% data.frame() %>% 
  dplyr::filter(P == 0.2, G == 25) %>%
  dplyr::select(c(Model, Train.Size, Batch.Size, MAE, RMSE, LCL)) %>%
  dplyr::mutate(MAE = round(MAE, 4), RMSE = round(RMSE, 4), LCL = round(LCL, 4))


data_filtered %>% gt() %>%
  gt::tab_spanner(label = md('**Autoencoder Parameters**'), columns = c(Model, Train.Size, Batch.Size)) %>%
  gt::tab_spanner(label = md('**Performance**'), columns = c(RMSE, MAE, LCL)) %>%
  gt::tab_header(title = md("**Simulation Performance**"), subtitle = paste0())








library(latex2exp)
library(interpTools)

set.seed(1)
X = interpTools::simXt(N = 100)$Xt
X = (X - min(X)) / (max(X) - min(X))
X0 = interpTools::simulateGaps(list(X), p = c(0.05), g = c(5), K = 1)[[1]]$p0.05$g5[[1]]

op <- par(mfrow = c(1,1), mar = c(5,5,4,5))
plot(X0, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$X_t^{(0)}$'), xaxt='n', yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)


XI = zoo::na.approx(X0)
plot(XI, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$Z_t^{(0)}$'), xaxt='n', yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)




X_train1 = XI; X_train1[50:54] = NA
X_train2 = XI; X_train2[78:82] = NA

op <- par(mfrow = c(2, 1), mar = c(3.8,5,1,5))
plot(X_train1, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$X_{train}^{(1)}$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)

plot(XI, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$Y_{train}^{(1)}$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)



plot(X_train2, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$X_{train}^{(maxIter)}$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)

plot(XI, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$Y_{train}^{(maxIter)}$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)




X_int = X + rnorm(100, 0, 0.1)
op <- par(mfrow = c(1,1), mar = c(5,5,4,5))
plot(X_int, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$X_t^{(0)}$ Prediction'), 
     xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)


X_int_final = ifelse(is.na(X0), X_int, X0)
plot(X_int_final, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$Z_t^{(1)}$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)









plot(X0, type = 'o', lwd = 0.5, pch = 16, cex = 0.5, 
     xlab = latex2exp::TeX('time ($t$)'), ylab = latex2exp::TeX('$Z_t$'), xaxt='n',  yaxt='n'); grid()
axis(1, at = seq(0, 100, by = 10), las=1)
axis(2, at = seq(0, 1, by = 0.5), las=1)
for (i in 1:30){
  set.seed(i)
  X_int_temp = X + rnorm(100, 0, 0.1)
  X_int_temp_final = ifelse(is.na(X0), X_int_temp, NA)
  lines(X_int_temp_final, type = 'p', cex = 1, pch = '-', col = 'red')
}

