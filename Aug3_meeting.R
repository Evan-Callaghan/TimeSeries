## LSTM Recurrent Neural Networks

## Loading libraries
library(tensorflow)
library(keras)
library(interpTools)


#####################################
## DEMO 1: LSTM with Sunspots Data ##
#####################################

## Reading sunspots data
sunspots = read.csv('Data/monthly-sunspots.csv')
sunspots = sunspots$Sunspots[1:1490]


## Defining some functions:
get_block <- function(x){
  mask = ifelse(is.na(x), NA, TRUE)
  Nlen <- length(mask)
  mask <- which(is.na(mask))
  # case: there are missing points
  if(length(mask) > 0) {
    diffs <- mask[-1] - mask[-length(mask)]
    diffs <- which(diffs > 1)
    # case: 1 gap only, possibly no gaps
    if(length(diffs)==0) {
      blocks <- matrix(data=0, nrow=1, ncol=3)
      blocks[1, 1:2] <- c(mask[1], mask[length(mask)])
    } else {
      blocks <- matrix(data=0,nrow=length(mask),ncol=3)
      blocks[1, 1] <- mask[1]
      blocks[1, 2] <- mask[diffs[1]]
      k <- 1
      for(j in 1:length(diffs)) {
        k <- k+1
        blocks[k, 1:2] <- c(mask[diffs[j]+1],mask[diffs[j+1]])
      }
      blocks[k,2] <- max(mask)
      blocks <- blocks[1:k, ]
    }
    blocks[,3] <- blocks[,2] - blocks[,1] + 1
    # checks to remove start/end of sequence
    if(blocks[1,1]==1) {
      blocks <- blocks[-1, ]
    }
    if(blocks[length(blocks[,1]),2]==Nlen) {
      blocks <- blocks[-length(blocks[,1]), ] 
    }
  } else {
    blocks <- NULL
  }
  return(blocks)
}
df_to_X_Y <-function(data, window){
  X = c(); Y = c()
  M = length(data) - window
  for (i in 1:M){
    row = data[i:(i+window-1)]
    label = data[(i+window)]
    X = c(X, row)
    Y = c(Y, label)
  }
  X = array(matrix(X, nrow = M, byrow = TRUE), dim = c(M, window, 1))
  Y = array(matrix(Y, nrow = M, byrow = TRUE), dim = c(M, 1))
  return(list(tf$constant(X), tf$constant(Y)))
}
gapped_df_to_X_Y <- function(X0, window){
  
  ## Defining helpful parameters
  BLOCK = get_block(X0)
  B = dim(BLOCK)[1] + 1
  N = length(X0)
  INDEX = 1
  
  ## Validating gaps 
  diff = c()
  for (i in 2:B){
    if (i == B){
      diff = c(diff, 1000 - BLOCK[i-1, 2] - 1)}
    else{
      diff = c(diff, BLOCK[i,1] - BLOCK[i-1, 2] - 1)}
  }
  to_skip = which(diff < (window + 1)) + 1
  
  ## Initializing arrays
  X = c(); Y = c(); 
  
  ## Looping through each section of data
  for (b in 1:B){
    
    ## If the current section is not long enough, skip to next iteration
    if (b %in% to_skip){print(b)}
    
    else {
      
      ## Setting end of looping section
      if (b == B){
        M = N - window - 1}
      else{
        M = BLOCK[b,1] - window - 1}
      
      ## Looping through index in section
      for (i in INDEX:M){
        row = X0[i:(i+window-1)]; X = c(X, row) ## Appending rows
        target = X0[i+window]; Y = c(Y, target)    ## Appending labels
      }
    }
    ## Updating INDEX
    if (b < B){
      INDEX = BLOCK[b,2] + 1}
  }
  rows = length(X) / window ## Setting number of rows
  print(sum(is.na(X))); print(sum(is.na(Y)))
  
  ## Formatting X and Y
  X = array(matrix(X, nrow = rows, byrow = TRUE), dim = c(rows, window, 1))
  Y = array(matrix(Y, nrow = rows, byrow = TRUE), dim = c(rows, 1)) 
  
  return(list(tf$constant(X), tf$constant(Y)))
}


## Plotting sunspots data
plot(1:1490, sunspots, type = 'l', lwd = 2, 
     main = 'Monthly Sunspots Data', xlab = 'Month', ylab = 'Sunspots'); grid()


## Properly formatting the data
WINDOW = 5
BATCH = 15
data = df_to_X_Y(sunspots, WINDOW)
X = data[[1]]
Y = data[[2]]

print(dim(X))
print(dim(Y))

print(X[1:3,,])
print(Y[1:3,])


## Creating training and validation sets
X_train = X[1:1200,,]; Y_train = Y[1:1200,]
X_valid = X[1201:1485,,]; Y_valid = Y[1201:1485,]

print(dim(X_train))
print(dim(Y_train))
print(dim(X_valid))
print(dim(Y_valid))


## Building the model
model = keras_model_sequential(name = 'Model') %>% 
  layer_lstm(64, stateful = FALSE, batch_input_shape = c(BATCH, WINDOW, 1)) %>%
  layer_dense(8, activation = 'relu') %>%
  layer_dense(1, activation = 'linear')

summary(model)

cp = tf$keras$callbacks$ModelCheckpoint('Callbacks/weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                                        save_best_only = TRUE)

model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                  loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')

model %>% fit(X_train, Y_train, validation_data = list(X_valid, Y_valid), verbose = 0,
              epochs = 50, batch_size = BATCH, callbacks = cp)


## Visualizing predictions 
train_preds = model %>% predict(X_train, batch_size = BATCH, verbose = 0)
valid_preds = model %>% predict(X_valid, batch_size = BATCH, verbose = 0)

plot(1:1200, Y_train, type = 'l', lwd = 2, xlim = c(1, 1485), main = 'Sunspots Data with LSTM Model', 
     ylab = 'Sunspots', xlab = 'Month'); grid()
lines(1201:1485, Y_valid, type = 'l', lwd = 2)
lines(1:1200, train_preds, type = 'l', col = 'red')
lines(1201:1485, valid_preds, type = 'l', col = 'dodgerblue')
legend(legend = c('Training', 'Validation'), lty = 1, lwd = 4, cex = 0.8,
       col = c('red', 'dodgerblue'), 'topleft', title = 'LSTM Predictions')











################################
## DEMO 2: LSTM as an Imputer ##
################################


## Simulating a time series from interpTools
set.seed(52)
X = interpTools::simXt(N = 1000, numTrend = 1, numFreq = 2)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.1, g = 10, K = 1)[[1]]$p0.1$g10[[1]]


## Plotting simulated data
plot(X, type = 'l', lty = 3); grid()
lines(X0, type = 'l', lwd = 2)


## Properly formatting the data
BATCH = 15
WINDOW = 5
data = gapped_df_to_X_Y(X0, WINDOW)
X = data[[1]]
Y = data[[2]]

print(dim(X))
print(dim(Y))

## Creating training and validation sets
X_train = X[1:690,,]; Y_train = Y[1:690]
X_valid = X[691:840,,]; Y_valid = Y[691:840]

dim(X_train)
dim(Y_train)
dim(X_valid)
dim(Y_valid)


## Building the model
model = keras_model_sequential(name = 'Model') %>% 
  layer_lstm(64, stateful = FALSE, batch_input_shape = c(BATCH, WINDOW, 1)) %>%
  layer_dense(8, activation = 'relu') %>%
  layer_dense(FORECAST, activation = 'linear')

summary(model)

cp = tf$keras$callbacks$ModelCheckpoint('Callbacks/weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                                        save_best_only = TRUE)

model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                  loss = 'MeanSquaredError', metrics = 'MeanAbsoluteError')

model %>% fit(X_train, Y_train, epochs = 50, batch_size = BATCH, verbose = 0,
              validation_data = list(X_valid, Y_valid), callbacks = cp)


## Visualizing predictions
train_preds = model %>% predict(X_train, batch_size = BATCH, verbose = 0)
valid_preds = model %>% predict(X_valid, batch_size = BATCH, verbose = 0)

plot(1:690, Y_train, type = 'l', lwd = 2, xlim = c(1, 840), main = 'Simulated Data with LSTM Model (Trend)', 
     ylab = 'X', xlab = 'Time'); grid()
lines(691:840, Y_valid, type = 'l', lwd = 2)
lines(1:690, train_preds, type = 'l', col = 'red')
lines(691:840, valid_preds, type = 'l', col = 'dodgerblue')
legend(legend = c('Training', 'Validation'), lty = 1, lwd = 4, cex = 0.8,
       col = c('red', 'dodgerblue'), 'topright', title = 'LSTM Predictions')



## LSTM cannot extrapolate... let's remove the initial trend component:

set.seed(52)
X = interpTools::simXt(N = 1000, numTrend = 1, numFreq = 2)$Xt
Mt = estimator(X, method = 'Mt')
X = X - Mt
X0 = interpTools::simulateGaps(list(X), p = 0.1, g = 10, K = 1)[[1]]$p0.1$g10[[1]]


## Plotting the simulated data
plot(X, type = 'l', lty = 3); grid()
lines(X0, type = 'l', lwd = 2)

## Properly formatting the data
data = gapped_df_to_X_Y(X0, WINDOW)
X = data[[1]]
Y = data[[2]]


## Creating training and validation sets
X_train = X[1:690,,]; Y_train = Y[1:690]
X_valid = X[691:840,,]; Y_valid = Y[691:840]


## Re-fitting the model
model %>% fit(X_train, Y_train, epochs = 50, batch_size = BATCH, verbose = 0,
              validation_data = list(X_valid, Y_valid), callbacks = cp)


## Visualizing predictions
train_preds = model %>% predict(X_train, batch_size = BATCH, verbose = 0)
valid_preds = model %>% predict(X_valid, batch_size = BATCH, verbose = 0)

plot(1:690, Y_train, type = 'l', lwd = 2, xlim = c(1, 840), main = 'Simulated Data with LSTM Model (No Trend)', 
     ylab = 'X', xlab = 'Time'); grid()
lines(691:840, Y_valid, type = 'l', lwd = 2)
lines(1:690, train_preds, type = 'l', col = 'red')
lines(691:840, valid_preds, type = 'l', col = 'dodgerblue')
legend(legend = c('Training', 'Validation'), lty = 1, lwd = 4, cex = 0.8,
       col = c('red', 'dodgerblue'), 'topright', title = 'LSTM Predictions')





