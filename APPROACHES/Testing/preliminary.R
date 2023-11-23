##########################################
## Autoencoder Preliminary Testing File ##
##########################################


#* Requirements:
#*    tensorflow version ...
#*    keras version ...
#*    Three data sets (as csv files)



## Importing libraries
## -----------------------

#library(parallel)
library(tensorflow)
library(keras)
library(interpTools)
#library(tictoc)
# install_tensorflow(envname = "r-tensorflow")


## Configuring GPU set-up
## -----------------------

## NOTE: R Project set up to use virtual Python version
gpus = tf$config$experimental$list_physical_devices('GPU')
# for (gpu in gpus){
#   tf$config$experimental$set_memory_growth(gpu, TRUE)
# }


# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)
# with strategy.scope():
#   inputs = tf.keras.layers.Input(shape=(1,))
# predictions = tf.keras.layers.Dense(1)(inputs)
# model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
# model.compile(loss='mse',
#               optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))


## Importing functions
## -----------------------

source('Simulations/Preliminary/Code/autoencoder_helper.R')


## Simulation Parameters
## -----------------------

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 5

TRAIN_SIZE = c(640, 1280, 2560)
EPOCHS = c(25, 50, 100)
BATCH_SIZE = c(16, 32, 64)



P = c(0.1)
G = c(5)
K = 5

TRAIN_SIZE = c(320, 640, 1280)
EPOCHS = c(25, 50, 75)
BATCH_SIZE = c(16, 32, 64)


## Reading the data sets
## -----------------------

sunspots = read.csv('Data/Cleaned/sunspots.csv')$sunspots
apple = read.csv('Data/Cleaned/apple.csv')$apple
temperature = read.csv('Data/Cleaned/toronto_temperature.csv')$temperature
high_snr = read.csv('Data/Cleaned/simulated.csv')$high_snr
low_snr = read.csv('Data/Cleaned/simulated.csv')$low_snr
modulated = read.csv('Data/Cleaned/simulated.csv')$modulated



## Simulation 1: Sunspots Data
## -----------------------

# Creating custom models for sunspots data
model1 = create_model(N = length(sunspots), units = 64, connected_units = 16, 'relu')
model2 = create_model(N = length(sunspots), units = 128, connected_units = 32, 'relu')
MODELS = c(model1, model2)

# Running the imputation simulation
sunspots_sim = simulation_main(sunspots, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)

# Exporting simulation performance as a csv file
write.csv(sunspots_sim, 'Simulations/Preliminary/Results/Prelim_November2023_sunspots100.csv', row.names = FALSE)

# Observed simulation time: ____



data1 = read.csv('Simulations/Preliminary/Results/Prelim_November2023_sunspots.csv')
data2 = read.csv('Simulations/Preliminary/Results/Prelim_November2023_sunspots2.csv')
data3 = read.csv('Simulations/Preliminary/Results/Prelim_November2023_sunspots3.csv')
data4 = read.csv('Simulations/Preliminary/Results/Prelim_November2023_sunspots4.csv')
data5 = read.csv('Simulations/Preliminary/Results/Prelim_November2023_sunspots5.csv')


head(data1)
head(data2)
head(data3)
head(data4)
head(data5)



## Simulation 2: Apple Stock Data
## -----------------------

# Creating custom models for apple data
model1 = create_model(N = length(apple), units = 64, connected_units = 32, 'relu')
model2 = create_model(N = length(apple), units = 128, connected_units = 32, 'relu')
MODELS = c(model1, model2)

# Running the imputation simulation
tic('Apple Simulation')
apple_sim = simulation_main(apple, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
toc()

# Exporting simulation performance as a csv file
write.csv(apple_sim, 'Simulations/Preliminary/Results/Prelim_November2023_apple.csv', row.names = FALSE)

# Observed simulation time: ____



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






