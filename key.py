import pandas as pd
import numpy as np
from scipy import stats
import scipy
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Configuing GPU setup: 
## -----------------------
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)


# Defining "Autoencoder" Functions: 
## -----------------------

def main(x0, max_iter, model_id, train_size, batch_size):
    
    # Defining matrix to store results
    results = np.zeros(shape = (max_iter, np.size(x0)))
    
    # Step 1: Estimating p and g
    p = estimate_p(x0); g = estimate_g(x0)
    
    # Step 2: Linear imputation
    xV = linear(x0)
    
    # Sanity Check
    print('P:', p, '   G:', g, '   Model:', model_id, '   train_size:', train_size, '   batch_size:', batch_size)
    
    for i in range(max_iter):
      
      # Steps 3/4: Simulating time series and imposing gap structure
      data = simulator(x0, xV, p, g, i, train_size)
      inputs = data[0]; targets = data[1]
      
      # Step 5: Performing the imputation
      preds = imputer(x0, inputs, targets, model_id, batch_size)
      
      # Step 6: Extracting the predicted values and updating imputed series
      xV = np.where(np.isnan(x0), preds, x0); results[i,:] = xV
      
      # Clearing intermediate results
      del data, inputs, targets
    
    # Returning a point estimation for each missing data point
    if (max_iter == 1):
        return results[0,]
    
    # Returning a distribution for each missing data point
    else:
        return results 

def estimate_p(x0):
    
    return np.round(np.sum(np.isnan(x0)), 2) / np.size(x0)

def estimate_g(x0):
    
    condition = True
    N = np.size(x0)
    i = 0
    
    widths = np.array([])
    
    while(condition):
        if (~np.isnan(x0[i])):
            cont = True
            start_idx = i
            while(cont):
                i = i + 1
                
                if (i >= N):
                    break
                    
                if ((~np.isnan(x0[i]))):
                    continue
                else:
                    end_idx = i - 1
                    cont = False
        
        else:
            cont = True
            start_idx = i
            while(cont):
                i = i + 1
                if ((np.isnan(x0[i])) & (i < N)):
                    continue
                else:
                    end_idx = i - 1
                    cont = False
                    
            widths = np.append(widths, [end_idx - start_idx + 1])
            
        if (i >= N):
            condition = False

    # Computing g
    g = math.floor(stats.mode(widths)[0].item())
    
    return g

def linear(x0):
    
    return np.array(pd.Series(x0).interpolate(method = 'linear'))

def simulator(x0, xV, p, g, iteration, train_size):
    
    N = np.size(xV)
    inputs = np.zeros((train_size, N, 1), dtype = 'float32')
    targets = np.zeros((train_size, N, 1), dtype = 'float32')
    
    for i in range(train_size):
        
        random.seed((iteration * train_size) + i)  # Setting a common seed
        
        inputs[i,:,0] = create_gaps(xV, x0, p, g)  # Appending inputs
        targets[i,:,0] = xV                        # Appending targets
    
    return inputs, targets

def create_gaps(x, x0, p, g):
    
    N = np.size(x)                           # Defining the number of data points
    to_remove = np.array([], dtype = 'int')  # Initializing vector to store removal indices
    
    # Creating list of possible indices to remove
    poss_values = np.where(~np.isnan(x0))
    poss_values = np.delete(poss_values, [-1, 0])
        
    # Determining number of data points to remove
    if ((p * N / g % 1 <= 0.5) & (p * N / g % 1 != 0)):
        end_while = math.floor(p * N) - g
    else:
        end_while = math.floor(p * N)
        
    # Deciding which indices to remove
    num_missing = 0
    iter_control = 0
    
    while(num_missing < end_while):
        
        start = np.random.choice(poss_values, 1)
        end = start + g
        
        if (np.isin(np.arange(start, end), poss_values).all()):
            
            poss_values = poss_values[~np.isin(poss_values, np.arange(start, end))]
            to_remove = np.append(to_remove, np.arange(start, end))
            num_missing = num_missing + g
            
        iter_control = iter_control + 1
        
        if (iter_control % 150 == 0):
            end_while = end_while - g
            
    # Placing NA in the indices to remove (represented by 0)
    to_return = x.copy()
    to_return[to_remove] = 0
    
    # Sanity check
    to_return[0] = x[0]
    to_return[N-1] = x[N-1]
        
    # Returning the final time series
    return to_return

def create_model(N, units, connected_units, activation):
    
    # Initializing model
    model = tf.keras.Sequential(name = 'Autoencoder')
    
    # Adding input layer
    model.add(tf.keras.layers.Input(shape = (N, 1), name = 'Input'))
    
    # Adding LSTM layer
    model.add(tf.keras.layers.LSTM(units, return_sequences = True, name = 'LSTM'))
    
    # Adding first dense layer
    model.add(tf.keras.layers.Dense(units, activation = activation, name = 'Encoder1'))
    
    # Adding encoder dense layers
    units_temp = units / 2; encoder_index = 2
    
    while(units_temp > connected_units):
        model.add(tf.keras.layers.Dense(units_temp, activation = activation, name = 'Encoder' + str(encoder_index)))
        units_temp = units_temp / 2; encoder_index = encoder_index + 1
        
    # Adding fully-connected layer
    model.add(tf.keras.layers.Dense(connected_units, activation = activation, name = 'Connected'))
    
    # Adding decoder dense layers
    units_temp = units_temp * 2; decoder_index = 1
    
    while(units_temp <= units):
        model.add(tf.keras.layers.Dense(units_temp, activation = activation, name = 'Decoder'+str(decoder_index)))
        units_temp = units_temp * 2; decoder_index = decoder_index + 1
        
    # Adding output layer
    model.add(tf.keras.layers.Dense(1, name = 'Output'))
    
    return model

def generate_model(N, model_id):
    
    if (model_id == 1):
        
        model = tf.keras.Sequential(name = 'Autoencoder')
        model.add(tf.keras.layers.Input(shape = (N, 1), name = 'Input'))
        model.add(tf.keras.layers.LSTM(64, return_sequences = True, name = 'LSTM'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'Encoder1'))
        model.add(tf.keras.layers.Dense(32, activation = 'relu', name = 'Encoder2'))
        model.add(tf.keras.layers.Dense(16, activation = 'relu', name = 'Connected'))
        model.add(tf.keras.layers.Dense(32, activation = 'relu', name = 'Decoder1'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'Decoder2'))
        model.add(tf.keras.layers.Dense(1, name = 'Output'))
        
    elif (model_id == 2):
        
        model = tf.keras.Sequential(name = 'Autoencoder')
        model.add(tf.keras.layers.Input(shape = (N, 1), name = 'Input'))
        model.add(tf.keras.layers.LSTM(128, return_sequences = True, name = 'LSTM'))
        model.add(tf.keras.layers.Dense(128, activation = 'relu', name = 'Encoder1'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'Encoder2'))
        model.add(tf.keras.layers.Dense(32, activation = 'relu', name = 'Connected'))
        model.add(tf.keras.layers.Dense(64, activation = 'relu', name = 'Decoder1'))
        model.add(tf.keras.layers.Dense(128, activation = 'relu', name = 'Decoder2'))
        model.add(tf.keras.layers.Dense(1, name = 'Output'))
        
    return model

def imputer(x0, inputs, targets, model_id, batch_size):
    
    # Storing series length
    N = np.shape(inputs)[1]
    
    # Storing inputs, targets, and x0 as tensors
    inputs = tf.constant(inputs); targets = tf.constant(targets)
    x0 = tf.constant(np.where(np.isnan(x0), 0, x0).reshape(1, N, 1))
    
    # Defining EarlyStopping callback
    callbacks = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10, restore_best_weights = 'True')
    
    # Defining the distributed strategy for model fitting
    strategy = tf.distribute.MirroredStrategy(devices = ['/gpu:0', '/gpu:1'])
    
    with strategy.scope():
      
      # Creating the model
      model = generate_model(N, model_id)
      
      # Compiling the model
      model.compile(loss = 'MeanSquaredError', optimizer = 'adam')
      
      # Fitting the model
      model.fit(inputs, targets, epochs = 100, batch_size = batch_size, shuffle = True, validation_split = 0, callbacks = [callbacks], verbose = 0)
    
    # Generating new model and copying trained model weights
    prediction_model = generate_model(N, model_id)
    prediction_model.set_weights(model.get_weights())  
    
    # Predicting on the original time series
    preds = prediction_model.predict(x0, verbose = 0)

    # Clearing the TensorFlow backend
    tf.keras.backend.clear_session()
    
    return preds[0,:,0]


# Defining Simulation Functions:
## -----------------------

def simulation(X, X0, MODELS, TRAIN_SIZE, BATCH_SIZE):
    
    # Setting common seed
    random.seed(42)
    
    # Defining helpful parameters
    N = X0.shape[0]; M = X0.shape[1]
    
    # Initializing DataFrame to store results
    results = pd.DataFrame()
    
    for m in range(M):
        for model in MODELS:
            for train_size in TRAIN_SIZE:
                for batch_size in BATCH_SIZE:

                    # Extracting X0 column for imputation
                    X0_temp = np.array(X0.iloc[:,m])

                    # Performing interpolation
                    interp = main(X0_temp, 1, model, train_size, batch_size)

                    # Performing performance assessment
                    performance = simulation_perf(np.array(X['data']), np.array(X0_temp), np.array(interp))

                    # Saving iteration results
                    results_iter = simulation_save(performance, np.array(X0_temp), model, train_size, batch_size)

                    # Appending iteration results to total results
                    results = pd.concat([results, results_iter], axis = 0)
                    
                    # Sanity Check: Saving progress 
                    results.to_csv('Simulations/Preliminary/Results/simulation_check_in.csv', index = False)
    
    # Resetting indices
    results = results.reset_index(drop = True)
    
    return results

def simulation_perf(X, X0, XI):
    
    # Identify indices of interpolated values
    idx = np.where(np.isnan(X0))
    
    # Only considering values which have been replaced
    X = X[idx]
    XI = XI[idx]
    
    # Defining helpful variables
    N = np.size(X)
    
    # Computing performance metrics:
    
    # Mean Absolute Error 
    MAE = np.sum(np.abs(np.subtract(XI, X))) / N
    
    # Root Mean Square Error
    RMSE = np.sqrt(np.sum(np.power(XI - X, 2)) / N)
    
    # Log-Cosh Loss
    LCL = np.sum(np.log(np.cosh(np.subtract(XI, X)))) / N
    
    return [MAE, RMSE, LCL]

def simulation_save(performance, X0, model, train_size, batch_size):
    
    # Estimating p and g
    p = estimate_p(X0); g = estimate_g(X0)
    
    # Creating results DataFrame
    results = pd.DataFrame({'P':p, 'G':g, 'Model':model, 'Train Size':train_size, 'Batch Size':batch_size, 
                            'MAE':performance[0], 'RMSE':performance[1], 'LCL':performance[2]}, index = [0])
    
    return results


## Simulation Parameters
## -----------------------

MODELS = [1]
TRAIN_SIZE = [500, 1000, 2000]
BATCH_SIZE = [16, 32, 64]


# Performing Simulations:
## -----------------------

# 1. Sunspots Data

# Reading time series data-frames
sunspots = pd.read_csv('Simulations/Preliminary/Data/sunspots_data.csv')
sunspots0 = pd.read_csv('Simulations/Preliminary/Data/sunspots_data0.csv')

sunspots.head()
sunspots0.head()

# Running the imputation simulation
sunspots_sim = simulation(sunspots, sunspots0, MODELS, TRAIN_SIZE, BATCH_SIZE)

# Exporting simulation performance as a csv file
sunspots_sim.to_csv('Simulations/Preliminary/Results/Prelim_sunspots.csv', index = False)

sunspots_sim.head()


# 2. Apple Data

# Reading time series data-frames
apple = pd.read_csv('Simulations/Preliminary/Data/apple_data.csv')
apple0 = pd.read_csv('Simulations/Preliminary/Data/apple_data0.csv')

apple.head()
apple0.head()

# Running the imputation simulation
apple_sim = simulation(apple, apple0, MODELS, TRAIN_SIZE, BATCH_SIZE)

# Exporting simulation performance as a csv file
apple_sim.to_csv('Simulations/Preliminary/Results/Prelim_apple.csv', index = False)

apple_sim.head()



# 3. Temperature Data

# Reading time series data-frames
temperature = pd.read_csv('Simulations/Preliminary/Data/temperature_data.csv')
temperature0 = pd.read_csv('Simulations/Preliminary/Data/temperature_data0.csv')

temperature.head()
temperature0.head()

# Running the imputation simulation
temperature_sim = simulation(temperature, temperature0, MODELS, TRAIN_SIZE, BATCH_SIZE)

# Exporting simulation performance as a csv file
temperature_sim.to_csv('Simulations/Preliminary/Results/Prelim_temperature.csv', index = False)

# Completed in ~__ hours occupying ~__ GB of RAM
temperature_sim.head()


View(temperature_sim)



# 4. High SNR Data

high_snr = pd.read_csv('Data/Exported/high_snr_data.csv')
high_snr0 = pd.read_csv('Data/Exported/high_snr_data0.csv')

high_snr.head()
high_snr0.head()

interp = main(high_snr0['0.3_25_1'], 1, 2, 640, 32)

fig = plt.figure(figsize = (12,4))
plt.plot(interp, color = 'red', linewidth = 0.3, label = 'Interpolation')
plt.plot(high_snr['data'], color = 'green', linewidth = 0.3, label = 'Original')
plt.plot(high_snr0['0.3_25_1'], color = 'black', label = 'Missing')
plt.legend(fontsize = 6, loc = 'upper right')
plt.grid()
plt.show()

simulation_perf(np.array(high_snr['data']), np.array(high_snr0['0.3_25_1']), np.array(interp))

models = [1, 2]
train_size = [320, 640, 1280]
batch_size = [16, 32, 64]



interp = simulation(high_snr, high_snr0, models, train_size, batch_size)

interp.head()

interp.to_csv('Data/Prelim_Autoencoder_November2023_high_snr.csv', index = False)






