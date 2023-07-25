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
  layer_masking(mask_value = 0, input_shape = c(9, 1), name = 'mask') %>%
  layer_lstm(100, input_shape = c(9, 1), activation = 'relu', name = 'Encoder') %>%
  layer_repeat_vector(9, name = 'Repeat') %>%
  layer_lstm(100, activation = 'relu', return_sequences = TRUE, name = 'Decoder') %>%
  time_distributed(layer_dense(units = 1), name = 'Final')

## Compiling the model
model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')

## Fitting the model to sequence
model %>% fit(sequence, sequence, epochs = 300, verbose = 0)

## Predicting on sequence
model %>% predict(sequence, verbose = 0)



## ----------------------------

## Defining the sequence
N = 100
sequence = interpTools::simXt(N = N, numTrend = 0, mu = 0)$Xt
X0 = interpTools::simulateGaps(list(sequence), p = 0.1, g = 5, K = 1)[[1]]$p0.1$g5[[1]]
X0 = ifelse(is.na(X0), 0, sequence)

plot(sequence, type = 'l', col = 'red'); grid()
lines(X0, type = 'l', lwd = 2)

## Reshaping sequence for LSTM layer
dim(X0) = c(1, N, 1)

## Converting vector to tensor
X0 = tf$constant(X0)

## Constructing the model
model = keras_model_sequential(name = 'Model') %>%  
  layer_masking(mask_value = 0, input_shape = c(N, 1), name = 'mask') %>%
  layer_lstm(100, input_shape = c(N, 1), activation = 'relu', name = 'Encoder') %>%
  layer_repeat_vector(N, name = 'Repeat') %>%
  layer_lstm(100, activation = 'relu', return_sequences = TRUE, name = 'Decoder') %>%
  time_distributed(layer_dense(units = 1, activation = 'linear'), name = 'Final')

## Compiling the model
model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')

## Fitting the model to sequence
model %>% fit(X0, X0, epochs = 100, verbose = 0)

## Predicting on sequence
preds = model %>% predict(X0, verbose = 0)

plot(X0, type = 'l', col = 'dodgerblue')
lines(preds, type = 'l', lwd = 2)



# define input sequence
sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat[0,:,0])