#################
## Autoencoder ##
#################


## Defining all functions
## -----------------------


#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#'
main <- function(x0, max_iter, model, train_size, epochs, batch_size){
  
  # Defining useful variables
  N = length(x0)
  
  # Defining matrix to store results
  results = matrix(NA, ncol = N, nrow = max_iter)
  
  ## Step 1: Estimating p and g
  p = estimate_p(x0); g = estimate_g(x0)
  
  ## Step 2: Linear imputation
  xV = zoo::na.approx(x0)
  
  for (i in 1:max_iter){
    
    ## Steps 3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xV, p, g, i, train_size)
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    preds = imputer(x0, inputs, targets, model, epochs, batch_size)
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), preds[1,,1], x0); results[i,] = xV
  }
  
  ## Returning a point estimation for each missing data point
  if (max_iter == 1){
    return(results[1,])
  }
  
  ## Returning a distribution for each missing data point
  else{
    return(results) 
  }
}


#' estimate_p
#' 
#' Function that returns the proportion of missing data in the input series. Simply 
#' returns the number of NA values divided by the length.
#' @param x0 {list}; List object containing the original incomplete time series
#'
estimate_p <- function(x0){
  N = length(x0)
  return(round(sum(is.na(x0))/N, 2))
}


#' estimate_g
#'
#' Function that returns the estimated width of missing data gaps in the input series.
#' Finds the length of all blocks of missing data and returns the mode of that list.
#' @param x0 {list}; List object containing the original incomplete time series
#'
estimate_g <- function(x0){
  
  ## Defining helpful parameters
  condition = TRUE
  N = length(x0); i = 1
  
  ## Initializing vector to store results
  widths = c()
  
  while(condition){
    
    if (!is.na(x0[i])){
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (!is.na(x0[i])){next}
        else{end_idx = i - 1; continue = FALSE}
      }
    }
    
    else{
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (is.na(x0[i]) & i <= N){next}
        else{end_idx = i - 1; continue = FALSE}
      }
      widths = c(widths, end_idx - start_idx + 1)
    }
    if (i > N){condition = FALSE}
  }
  
  ## Computing g
  u_widths = unique(widths)
  g = u_widths[which.max(tabulate(match(widths, u_widths)))]
  
  return(g)
}



#' simulator
#' 
#' Function to generate a set of training data for the network by randomly imposing 
#' the specified gap structure into the original incomplete time series.
#' 
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param p {float}; Proportion of data to be removed
#' @param g {integer}; Width of missing data sections
#' @param iteration {integer}; Current iteration of the autoencoder algorithm
#' @param train_size {integer}; Number of new time series to construct
#' 
simulator <- function(x0, xV, p, g, iteration, train_size){
  
  N = length(xV); n_missing = N * p ## Defining useful parameters
  inputs_temp = c(); targets_temp = c(); weights_temp = c() ## Initializing vectors to store values
  
  for (i in 1:train_size){
    
    set.seed((iteration * train_size) + i) ## Setting a common seed
    x_g = create_gaps(xV, x0, p, g) ## Imposing randomized gap structure
    
    inputs_temp = c(inputs_temp, x_g) ## Appending inputs
    targets_temp = c(targets_temp, xV) ## Appending targets
  }
  
  inputs = array(matrix(inputs_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting targets
  return(list(inputs, targets))
}


#' create_gaps
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. This function 
#' simulates the MCAR process to impose gaps into a time series according to a set combination of missing proportion and 
#' gap width. What differentiates this from Sophie's implementation us that the returned gapped series does not allow for gaps
#' to be placed where the original incomplete time series (x0) has missing data. This is for specific application in the 
#' autoencoder method for time series data imputation.
#' @param x {list}; List containing the complete time series to be imposed with missing values (equivalent to "x version")
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param p {float}; Proportion of data to be removed
#' @param g {integer}; Width of missing data sections
#' 
create_gaps <- function(x, x0, p, g){
  
  n = length(x) ## Defining the number of data points
  to_remove = c() ## Initializing vector to store removal indices
  
  ## Some checks:
  stopifnot(is.numeric(x), is.numeric(x0), is.numeric(p), is.numeric(g), !is.null(x), !is.null(x0), !is.null(p), !is.null(g), 
            sum(is.na(x)) == 0, g %% 1 == 0, length(x) > 2,  p >= 0 & p <= (n-2)/n, g >= 0, p*g < length(x)-2)
  
  ## Creating list of possible indices to remove
  poss_values = which(!is.na(x0))
  if (1 %in% poss_values){poss_values = poss_values[-1]}
  if (n %in% poss_values){poss_values = poss_values[-length(poss_values)]}
  
  ## Determining number of data points to remove
  if ((p * n / g) %% 1 != 0) {
    warning(paste("Rounded to the nearest integer multiple; removed ", round(p*n/g,0)*g, " observations", sep =""))
  }
  
  if((p * n / g) %% 1 <= 0.5 & (p * n / g) %% 1 != 0) {
    end_while <- floor(p * n) - g
  } else {
    end_while <- floor(p * n)
  }
  
  ## Deciding which indices to remove
  num_missing = 0
  iter_control = 0
  
  while(num_missing < end_while) {
    
    start = sample(poss_values, 1)
    end = start + g - 1
    
    if (all(start:end %in% poss_values)){
      poss_values = poss_values[!poss_values %in% start:end]
      to_remove = c(to_remove, start:end)
      num_missing = num_missing + g
    }
    
    iter_control = iter_control + 1
    
    if (iter_control %% 150 == 0){
      end_while = end_while - g
    }
  }
  
  ## Placing NA in the indices to remove (represented by 0)
  x.gaps = x
  x.gaps[to_remove] = 0
  
  ## Sanity check
  x.gaps[1] = x[1]
  x.gaps[n] = x[n]
  
  ## Returning the final gappy data
  return(as.numeric(x.gaps))
}


#' impute
#' 
#' Function to construct, compile, and fit a neural network model on the simulated training data
#' and then predict on the original time series. Completes the fifth step of the designed algorithm.
#' @param x0 {list}; List containing the original incomplete time series ("x naught")
#' @param inputs {matrix}; Matrix object containing the input training data
#' @param targets {matrix}; Matrix object containing the target training data
#' @param model {model}; Model architecture to be used for the neural network
#' 
imputer <- function(x0, inputs, targets, model, epochs, batch_size){
  
  # Defining useful parameters
  N = dim(inputs)[2]
  
  # Re-formatting original series
  x0 = matrix(ifelse(is.na(x0), 0, x0), ncol = 1); dim(x0) = c(1, N, 1)
  
  inputs = tf$constant(inputs) # Creating input tensors
  targets = tf$constant(targets) # Creating target tensors
  x0 = tf$constant(x0) # Creating prediction tensor
  
  # Compiling the model
  #model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')
  
  # Fitting the model to the training data
  model %>% fit(inputs, targets, epochs = epochs, batch_size = batch_size, 
                shuffle = FALSE, validation_split = 0.2, verbose = 0)
  
  # Predicting on the original series
  preds = model %>% predict(x0, verbose = 0)
  return(preds)
}


#' get_model
#' 
#' Function to return the desired TensorFlow neural network architecture.
#' @param N {integer}; Length of the original time series
#' 
get_model <- function(N){
  
  layer_1 = layer_input(shape = c(N, 1), batch_shape = NULL, name = 'Input')
  layer_2 = layer_lstm(units = 128, return_sequences = TRUE, name = 'LSTM')
  layer_3 = layer_dense(units = 128, activation = 'relu', name = 'Encoder1')
  layer_4 = layer_dense(units = 64, activation = 'relu', name = 'Encoder2')
  layer_5 = layer_dense(units = 32, activation = 'relu', name = 'Connected')
  layer_6 = layer_dense(units = 64, activation = 'relu', name = 'Decoder1')
  layer_7 = layer_dense(units = 128, activation = 'relu', name = 'Decoder2')
  layer_8 = layer_dense(units = 1, name = 'Output')
  
  autoencoder = keras_model_sequential(layers = c(layer_1, layer_2, layer_3, layer_4, 
                                                  layer_5, layer_6, layer_7, layer_8))
  return(autoencoder)
}


#' create_model
#' 
#' Function to create a neural network model with a customized architecture from TensorFlow.
#' 
#' @param N {integer}; Length of the time series for imputation
#' @param units {int}; Number of units for the LSTM layer and first dense layer
#' @param connected_units {int}; Number of units for the fully-connected layer
#' @param activation {string}; Activation functions to be used in each dense layer
#' 
create_model <- function(N, units, connected_units, activation){
  
  if (units %% connected_units != 0) stop('Connected units must be a factor of units.')
  
  # Creating input layer
  input = layer_input(shape = c(N, 1), batch_shape = NULL, name = 'Input')
  
  # Creating LSTM layer
  lstm = layer_lstm(units = units, return_sequences = TRUE, name = 'LSTM')
  
  # Creating first dense layer
  layers = c(layer_dense(units = units, activation = activation, name = 'Encoder1'))
  
  # Adding encoder dense layers
  units_temp = units / 2; encoder_index = 2
  
  while (units_temp > connected_units){
    layers = c(layers, layer_dense(units = units_temp, activation = activation, name = paste0('Encoder', encoder_index)))
    units_temp = units_temp / 2; encoder_index = encoder_index + 1
  }
  
  # Adding fully-connected layer
  layers = c(layers, layer_dense(units = connected_units, activation = activation, name = 'Connected'))
  
  # Adding decoder dense layers
  units_temp = units_temp * 2; decoder_index = 1
  
  while (units_temp <= units){
    layers = c(layers, layer_dense(units = units_temp, activation = activation, name = paste0('Decoder', decoder_index)))
    units_temp = units_temp * 2; decoder_index = decoder_index + 1
  }
  
  # Creating output layer
  output = layer_dense(units = 1, name = 'Output')
  
  # Forming and compiling the final model
  model = keras_model_sequential(layers = c(input, lstm, layers, output), name = 'Autoencoder') %>% 
    compile(optimizer = 'adam', loss = 'MeanSquaredError')
  
  return(model)
}


#' simulation_main
#' 
#' Function which facilitates the testing of the Neural Network Imputer (NNI) versus other methods implemented in the
#' interpTools package.
#' @param X {list}; List object containing a complete time series (should be scaled to (0,1))
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
simulation_main <- function(X, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE){
  
  # Setting common seed
  set.seed(42)
  
  # Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K); print('Imposed Gaps.')
  
  # Impute
  xI = simulation_impute(x0, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
  
  # Evaluate
  performance = simulation_performance(X, xI, x0)
  
  # Save
  results = simulation_saver(performance, xI, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
  
  # Return
  return(results)
}


#' simulation_impute
#' 
#' Function which acts as a wrapper to the interpTools interpolation process and also adds the ability to use the Neural 
#' Network Imputer (NNI). 
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#'
#'
#'
#'
#'
#'
#'
#'

simulation_impute <- function(x0, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE){
  
  # Defining helpful variables
  M = length(MODELS)
  TS = length(TRAIN_SIZE)
  E = length(EPOCHS)
  BS = length(BATCH_SIZE)
  iter = 1
  
  # Initializing lists
  batchsizeList <- list()
  epochsList <- list()
  trainsizeList <- list()
  samples <- list()
  
  # Creating naming conventions
  batchsize_names = paste0('bs', BATCH_SIZE)
  epochs_names = paste0('e', EPOCHS)
  trainsize_names = paste0('ts', TRAIN_SIZE)
  model_names = paste0('model', seq(1:length(MODELS)))
  
  # Defining the function call
  function_call = paste0('main(x, max_iter = 1, MODELS[[m]], TRAIN_SIZE[TS], EPOCHS[e], BATCH_SIZE[bs])')
  
  # Performing imputation
  for (m in 1:M){
    for (ts in 1:TS){
      for (e in 1:E){
        for (bs in 1:BS){
          
          # Applying the function call across all P, G, and K
          samples[[bs]] = lapply(x0[[1]], function(x){
            lapply(x, function(x){
              lapply(x, function(x){
                eval(parse(text = function_call))})})})
          
        }
        names(samples) <- batchsize_names
        batchsizeList[[e]] <- samples
      }
      names(batchsizeList) <- epochs_names
      epochsList[[ts]] <- batchsizeList
    }
    names(epochsList) <- trainsize_names
    trainsizeList[[m]] <- epochsList
  }
  names(trainsizeList) <- model_names
  
  return(trainsizeList)
}




#' simulation_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to compute imputation performance with a variety of loss functions / performance metrics.
#' @param X {list}; List object containing the original complete time series
#' @param xI {list}; List object containing the interpolated time series
#' @param x0 {list}; List object containing the original incomplete time series
#'
simulation_performance <- function(x, xI, x0){
  
  # Defining helpful variables
  M <- length(xI)
  TS <- length(xI[[1]])
  E <- length(xI[[1]][[1]])
  BS <- length(xI[[1]][[1]][[1]])
  P <- length(xI[[1]][[1]][[1]][[1]])
  G <- length(xI[[1]][[1]][[1]][[1]][[1]])
  K <- length(xI[[1]][[1]][[1]][[1]][[1]][[1]])
  
  # Initializing nested lists
  pf <- lapply(pf <- vector(mode = 'list', M), function(x)
    lapply(pf <- vector(mode = 'list', TS), function(x) 
      lapply(pf <- vector(mode = 'list', E), function(x) 
        lapply(pf <- vector(mode = 'list', BS), function(x)
          lapply(pf <- vector(mode = 'list', P), function(x)
            lapply(pf <- vector(mode = 'list', G), function(x)
              x <- vector(mode = 'list', K)))))))
  
  # Defining the naming conventions
  gap_names <- names(xI[[1]][[1]][[1]][[1]][[1]])
  prop_names <- names(xI[[1]][[1]][[1]][[1]])
  bs_names <- names(xI[[1]][[1]][[1]])
  epoch_names <- names(xI[[1]][[1]])
  ts_names <- names(xI[[1]])
  model_names <- names(xI)
  
  # Computing performance criteria
  for (m in 1:M){
    for (ts in 1:TS){
      for (e in 1:E){
        for (bs in 1:BS){
          for (p in 1:P){
            for (g in 1:G){
              for (k in 1:K){
                
                # Defining the temporary x0 and xI vectors
                x0_temp = x0[[1]][[p]][[g]][[k]]
                xI_temp = xI[[m]][[ts]][[e]][[bs]][[p]][[g]][[k]]
                
                # Computing performance metrics
                pf[[m]][[ts]][[e]][[bs]][[p]][[g]][[k]] <- unlist(simulation_performance_helper(x, xI_temp, x0_temp))
              }
              names(pf[[m]][[ts]][[e]][[bs]][[p]]) <- gap_names
            }
            names(pf[[m]][[ts]][[e]][[bs]]) <- prop_names
          }
          names(pf[[m]][[ts]][[e]]) <- bs_names
        }
        names(pf[[m]][[ts]]) <- epoch_names
      }
      names(pf[[m]]) <- ts_names
    }
    names(pf) <- model_names
  }
  
  # Selecting the class of pf
  class(pf) <- "pf"
  
  return(pf) 
}


#' simulation_performance_helper
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to aggregate imputer performance over several iterations and combinations of P, G, and K and return
#' a neatly laid out performance report.
#' @param x {list}; List object containing the original complete time series
#' @param X {list}; List object containing the interpolated time series
#' @param GappyData {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
#'
#'
#'
#'
#'
#'
simulation_performance_helper <- function(x, xI, x0) {
  
  # Identify indices of interpolated values
  index <- which(is.na(x0))
  
  # Only considering values which have been replaced
  xI <- xI[index]
  x <- x[index]
  
  # Defining helpful variables
  N <- length(x)
  
  # Initializing a list to store results
  return <- list()
  
  # Computing performance metrics:
  
  # Coefficient of Correlation, r
  numerator <- sum((xI - mean(xI))*(x - mean(x)))
  denominator <- sqrt(sum((xI - mean(xI))^2)) * sqrt(sum((x - mean(x))^2))
  return$pearson_r <- numerator / denominator
  
  # r^2
  return$r_squared <- return$pearson_r^2  
  
  # Absolute Differences
  return$AD <- sum(abs(xI - x))
  
  # Mean Bias Error 
  return$MBE <- sum(xI - x) / N
  
  # Mean Error 
  return$ME <- sum(x - xI) / N
  
  # Mean Absolute Error 
  return$MAE <- abs(sum(x - xI)) / length(x)
  
  # Mean Relative Error 
  if (length(which(x == 0)) == 0) {
    return$MRE <- sum((x - xI) / x)  
  } else {
    return$MRE <- NA
  }
  
  # Mean Absolute Relative Error ##### Lepot
  if (length(which(x == 0)) == 0) {
    return$MARE <- 1/length(x)*sum(abs((x - xI) / x))
  } else {
    return$MARE <- NA 
  }
  
  # Mean Absolute Percentage Error 
  return$MAPE <- 100 * return$MARE
  
  # Sum of Squared Errors
  return$SSE <- sum((xI - x)^2)
  
  # Mean Square Error 
  return$MSE <- 1 / N * return$SSE
  
  # Root Mean Squares, or Root Mean Square Errors of Prediction 
  if (length(which(x == 0)) == 0) {
    return$RMS <- sqrt(1 / N * sum(((xI - x)/x)^2))
  } else {
    return$RMS <- NA 
  }
  
  # Mean Squares Error (different from MSE, referred to as NMSE)
  return$NMSE <- sum((x - xI)^2) / sum((x - mean(x))^2)
  
  # Reduction of Error, also known as Nash-Sutcliffe coefficient 
  return$RE <- 1 - return$NMSE
  
  # Root Mean Square Error, also known as Root Mean Square Deviations
  return$RMSE <- sqrt(return$MSE)
  
  # Normalized Root Mean Square Deviations 
  return$NRMSD <- 100 * (return$RMSE / (max(x) - min(x)))
  
  # Root Mean Square Standardized Error 
  if (sd(x) != 0) {
    return$RMSS <- sqrt(1 / N * sum(((xI - x) / sd(x) )^2))  
  } else {
    return$RMSS <- NA 
  }
  
  # Median Absolute Percentage Error
  if (length(which(x == 0)) == 0) {
    return$MdAPE <- median(abs((x - xI) / x))*100  
  } else {
    return$MdAPE <- NA
  }
  
  # Log-Cosh Loss
  return$LCL <- sum(log(cosh(xI - x))) / N
  
  return(return)
}


#' simulation_saver
#' 
#' Function to store all simulation results in a consistent format. Takes the simulation performance as input
#' and sorts through the nested lists to organize imputation performance in a data-frame object. The returned 
#' data-frame can be easily exported to save simulation results.
#' @param performance {list}; Nested list object containing performance evaluation from the the set of simulations
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
simulation_saver <- function(performance, xI, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE){
  
  # Defining helpful variables
  M <- length(xI)
  TS <- length(xI[[1]])
  E <- length(xI[[1]][[1]])
  BS <- length(xI[[1]][[1]][[1]])
  
  # Creating naming conventions
  bs_names <- names(xI[[1]][[1]][[1]])
  epoch_names <- names(xI[[1]][[1]])
  ts_names <- names(xI[[1]])
  model_names <- names(xI)
  
  # Initializing data-frame to store results
  performance_summary = data.frame()
  
  # Looping through all simulation combinations
  for (m in 1:M){
    for (ts in 1:TS){
      for (e in 1:E){
        for (bs in 1:BS){
          for (p in P){
            for (g in G){
              for (k in 1:K){
                
                # Appending the selected performance metrics
                metrics = paste0("performance$", model_names[m], "$", ts_names[ts], "$", epoch_names[e], 
                                 "$", bs_names[bs], "$p", p, "$g", g, "[[", k, ']]')
                
                performance_summary = rbind(performance_summary, c(model_names[m], TRAIN_SIZE[ts], EPOCHS[e], BATCH_SIZE[bs], p, g, k, 
                                                                   as.numeric(eval(parse(text = metrics)))))
              }
            }
          }
        }
      }
    }
  }
  
  # Cleaning the final data-frame
  colnames(performance_summary) = c('Model', 'Train Size', 'Epochs', 'Batch Size', 'P', 'G', 'K', 'pearson_r', 'r_squared', 'AD', 'MBE', 'ME', 
                                    'MAE', 'MRE', 'MARE', 'MAPE', 'SSE', 'MSE', 'RMS', 'NMSE', 'RE', 'RMSE', 'NRMSD', 'RMSS', 'MdAPE', 'LCL')
  return(performance_summary)
}








## Testing the above functions
## -----------------------

library(tensorflow)
library(interpTools)
library(keras)

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}



# Single Interpolation:

N = 1500
X = interpTools::simXt(N)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.1, g = 10, K = 1)[[1]]$p0.1$g10[[1]]


model = create_model(N, units = 128, connected_units = 32, activation = 'relu')

int1 = main(X0, max_iter = 1, train_size = 320, model = model, epochs = 25, batch_size = 32)

int2 = main(X0, max_iter = 1, train_size = 2560, model = model, epochs = 100, batch_size = 32)


par(mfrow = c(2, 1), mai = c(0.5, 1, 0.5, 1))
plot(int1, type = 'l', col = 'green', main = 'Train Size = 320, Epochs = 25'); grid()
lines(X0, lwd = 1.2)

plot(int2, type = 'l', col = 'red', main = 'Train Size = 2560, Epochs = 100'); grid()
lines(X0, lwd = 1.2)





# Series of Interpolations:

X = interpTools::simXt(N = 1000)$Xt

P = c(0.1, 0.2)
G = c(10)
K = 3

model1 = create_model(N = 1000, units = 256, connected_units = 32, 'relu')
model2 = create_model(N = 1000, units = 128, connected_units = 32, 'relu')

MODELS = c(model1, model2)
TRAIN_SIZE = c(640)
EPOCHS = c(25, 50)
BATCH_SIZE = c(32)

tester = simulation_main(X, P, G, K, MODELS, TRAIN_SIZE, EPOCHS, BATCH_SIZE)
View(tester)







