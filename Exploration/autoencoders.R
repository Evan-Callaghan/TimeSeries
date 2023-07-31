library(tensorflow)
library(keras)

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}

## ----------------------------

## Defining the sequence
sequence = c(0.1, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

## Reshaping sequence for LSTM layer

#' LSTM networks require input data of shape: n_samples x time steps x n_features
dim(sequence) = c(1, 9, 1)
N_in = length(sequence)

## Converting vector to tensor
sequence = tf$constant(sequence)

## Constructing the model
model = keras_model_sequential(name = 'Model') %>%  
  layer_lstm(128, activation = 'relu', input_shape = c(9, 1), return_sequences = TRUE, name = 'Encoder1') %>%
  layer_lstm(64, activation = 'relu', name = 'Encoder2') %>%
  layer_repeat_vector(9, name = 'Repeat') %>%
  layer_lstm(100, activation = 'relu', return_sequences = TRUE, name = 'Decoder') %>%
  time_distributed(layer_dense(units = 1), name = 'Final')

## Compiling the model
model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')

## Fitting the model to sequence
model %>% fit(sequence, sequence, epochs = 100, verbose = 1)

## Predicting on sequence
model %>% predict(sequence, verbose = 0)








## ----------------------------

## Defining the sequence
N = 100
sequence = interpTools::simXt(N = N, numTrend = 1, mu = 0, numFreq = 2)$Xt
X0 = interpTools::simulateGaps(list(sequence), p = 0.1, g = 5, K = 1)[[1]]$p0.1$g5[[1]]
X0 = ifelse(is.na(X0), 0, sequence)

plot(sequence, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)

## Reshaping sequence for LSTM layer
dim(sequence) = c(1, N, 1)

## Converting vector to tensor
sequence = tf$constant(sequence)

## Constructing the model
model = keras_model_sequential(name = 'Model') %>%  
  layer_lstm(128, activation = 'relu', name = 'Encoder') %>%
  layer_repeat_vector(N, name = 'Repeat') %>%
  layer_lstm(128, activation = 'relu', return_sequences = TRUE, name = 'Decoder') %>%
  time_distributed(layer_dense(units = 1), name = 'Final')

## Compiling the model
model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')

## Fitting the model to sequence
model %>% fit(sequence, sequence, epochs = 500, verbose = 1)

## Predicting on sequence
preds = model %>% predict(sequence, verbose = 0)

plot(sequence, type = 'l', lwd = 2); grid()
lines(preds, type = 'l', col = 'dodgerblue')


#' Let's try adjusting the LSTM layers... add more and make this network
#' a little bit more complex. 




## ----------------------------

set.seed(26)
series = interpTools::simXt(N = 500, numTrend = 1)$Xt
plot(series, type = 'l', lwd = 2); grid()

mt = estimateMt(series, N = length(series), nw=5, k=8, pMax=3)
tt = estimateTt(series - mt, epsilon=1e-6, dT=1, nw=5, k=8,sigClip=0.999, progress=FALSE)
freqRet <- attr(tt, "Frequency")
if(length(freqRet) > 1 | (length(freqRet)==1 && freqRet != 0)) {
  TtP <- rowSums(tt) 
} else {
  TtP <- tt
}

lines(mt, type = 'l', col = 'red')
lines(TtP + mt, type = 'l', col = 'dodgerblue')









## ----------------------------

## LSTM FORECASTING:


## Function to create X and Y
df_to_x_y <- function(data, window_size = 5){
  X = c(); Y = c(); M = (length(data) - window_size)
  for (i in 1:M){
    row = data[i:(i+window_size-1)]
    label = data[i+window_size]
    X = c(X, row)
    Y = c(Y, label)
  }
  
  X = array(matrix(X, nrow = M, byrow = TRUE), dim = c(M, window_size))
  Y = array(matrix(Y, nrow = M, byrow = TRUE), dim = c(M, 1)) 
  return(list(X, Y))
}

WINDOW_SIZE = 5
data = df_to_x_y(sunspots, WINDOW_SIZE)

X = data[[1]]
Y = data[[2]]

dim(X)
dim(Y)

X_train = X[1:2000, 1:WINDOW_SIZE]; Y_train = Y[1:2000, 1]
X_valid = X[2001:2400, 1:WINDOW_SIZE]; Y_valid = Y[2001:2400, 1]
X_test = X[2401:2810, 1:WINDOW_SIZE]; Y_test = Y[2401:2810, 1]


dim(X_train)
dim(X_valid)
dim(X_test)


model1 = keras_model_sequential(name = 'Model') %>% 
  layer_lstm(64, input_shape = c(WINDOW_SIZE, 1)) %>%
  layer_dense(8, activation = 'relu') %>%
  layer_dense(1)
summary(model1)

cp = tf$keras$callbacks$ModelCheckpoint('model1/', save_best_only = TRUE)

model1 %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                   loss = 'MeanSquaredError')

model1 %>% fit(X_train, Y_train, validation_data = list(X_valid, Y_valid), 
               epochs = 25, callbacks = cp)

preds = model1 %>% predict(X_test)
plot(Y_test, type = 'l', lwd = 2); grid()
lines(preds, type = 'l', col = 'red')


preds_valid = model1 %>% predict(X_valid)
plot(Y_valid, type = 'l', lwd = 2); grid()
lines(preds_valid, type = 'l', col = 'red')


preds_train = model1 %>% predict(X_train)
plot(Y_train, type = 'l', lwd = 2); grid()
lines(preds_train, type = 'l', col = 'red')





## ----------------------------

## LSTM IMPUTER:

options(warn = -1)
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



sunspots = sunspots[1:1000]
sunspots0 = sunspots; sunspots0[475:525] = NA

plot(sunspots, type = 'l', col = 'red'); grid()
lines(sunspots0, type = 'l', lwd = 2)

blocks = get_block(sunspots0)
blocks

## Split data according to block structure
section1 = sunspots0[1:(blocks[1,1]-1)]
section2 = sunspots0[(blocks[1,2]+1):length(sunspots0)]


## Creating training sets
df_to_x_y_new <- function(data, window_length = 5, forecast_length = 1){
  X = c(); Y = c()
  M = length(data) - window_length #- forecast_length
  P = window_length + forecast_length
  
  ## Forward pass:
  for (i in 1:M){
    row = data[i:(i+window_length-1)]
    label = data[(i+window_length):(i+P)]
    X = c(X, row)
    Y = c(Y, label)
  }
  
  ## Backward pass:
  data = rev(data)
  for (i in 1:M){
    row = data[i:(i+window_length-1)]
    label = data[(i+window_length):(i+P)]
    X = c(X, row)
    Y = c(Y, label)
  }
  
  X = array(matrix(X, nrow = 2*M, byrow = TRUE), dim = c(2*M, window_length))
  Y = array(matrix(Y, nrow = 2*M, byrow = TRUE), dim = c(2*M, forecast_length)) 
  return(list(X, Y))
}


## Defining window size and forecast length
WINDOW_LENGTH = 10
FORECAST_LENGTH = 1

input_set1 = df_to_x_y_new(section1, WINDOW_LENGTH, FORECAST_LENGTH) 
X1 = input_set1[[1]]
Y1 = input_set1[[2]]

input_set2 = df_to_x_y_new(section2, WINDOW_LENGTH, FORECAST_LENGTH) 
X2 = input_set2[[1]]
Y2 = input_set2[[2]]


X_train = tf$constant(X); Y_train = tf$constant(Y)

dim(X_train)
dim(Y_train)


model = keras_model_sequential(name = 'Model') %>% 
  layer_lstm(64, input_shape = c(WINDOW_LENGTH, 1)) %>%
  layer_dense(16, activation = 'relu') %>%
  layer_dense(FORECAST_LENGTH, activation = 'linear')

summary(model)

cp = tf$keras$callbacks$ModelCheckpoint('Callbacks/weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                                        save_best_only = TRUE)

model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                  loss = 'MeanSquaredError')

# model %>% fit(X_train, Y_train, validation_data = list(X_valid, Y_valid), epochs = 25, callbacks = cp, verbose = 0)
model %>% fit(X_train, Y_train, validation_split = 0.2, epochs = 25, 
              callbacks = cp, verbose = 0)

sunspots0[474:526]




## Setting loop length
N = blocks[1, 3]

## Initializing vectors
start = blocks[1,1] - WINDOW_LENGTH; end = blocks[1,1] - FORECAST_LENGTH
to_predict = tf$constant(sunspots0[start:end])
preds = c()

for (i in 1:N){
  
  ## Expanding dimensions
  to_predict = tf$expand_dims(to_predict, 0L)
  
  ## Predicting on input vector
  prediction = model %>% predict(to_predict, verbose = 0)
  
  ## Saving prediction
  preds = c(preds, prediction)
  
  ## Updating prediction vector
  to_predict = tf$concat(list(to_predict, prediction), axis = 1L)[1][c(2:11)]
}

plot(sunspots0, type = 'l', lwd = 2); grid()
lines(blocks[1,1]:blocks[1,2], preds, col = 'dodgerblue')




preds = model %>% predict(X_train)
plot(Y_train, type = 'l', lwd = 2); grid()
lines(preds, type = 'l', col = 'red')



preds = model %>% predict(X_train)
plot(Y_train, type = 'l', lwd = 2); grid()
lines(preds, type = 'l', col = 'red')














