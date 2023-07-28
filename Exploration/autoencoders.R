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

data = df_to_x_y(sunspots, 5)

X = data[[1]]
Y = data[[2]]

dim(X)
dim(Y)

X_train = X[1:2000, 1:5]; Y_train = Y[1:2000, 1]
X_valid = X[2001:2400, 1:5]; Y_valid = Y[2001:2400, 1]
X_test = X[2401:2815, 1:5]; Y_test = Y[2401:2815, 1]


dim(X_train)
dim(X_valid)
dim(X_test)


model = keras_model_sequential(name = 'Model') %>% 
  layer_lstm(64, input_shape = c(5, 1)) %>%
  layer_dense(8, activation = 'relu') %>%
  layer_dense(1, activation = 'linear')
summary(model)

model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), 
                  loss = 'MeanSquaredError')

model %>% fit(X_train, Y_train, validation_data = list(X_valid, Y_valid), epochs = 10)

preds = model %>% predict(X_test)

plot(Y_test, type = 'l', lwd = 2); grid()
lines(preds, type = 'l', col = 'red')


