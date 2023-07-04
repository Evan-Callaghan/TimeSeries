library(reticulate)
path_to_python = install_python()
virtualenv_create("r-reticulate", python = path_to_python)

library(tensorflow)
install_tensorflow(envname = "r-reticulate")


library(keras)


fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

dim(train_images)
dim(train_labels)


library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")


train_images <- train_images / 255
test_images <- test_images / 255


par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))}


model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')




model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)



model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)


score <- model %>% evaluate(test_images, test_labels, verbose = 0)
cat('Test loss:', score["loss"], "\n")
cat('Test accuracy:', score["accuracy"], "\n")





predictions <- model %>% predict(test_images)



par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}








## -------------------------------------------------------

###########################
## Intro to Autoencoders ##
###########################


## Block to install necessary packages
library(remotes)
remotes::install_github('rstudio/tensorflow')
library(tensorflow)
install_tensorflow()
library(tensorflow)
tf_config()

## Loading MNIST image data set
mnist = dataset_mnist()

## Defining training and testing data
X_train = mnist$train$x
X_test = mnist$test$x

## Scaling all pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

## Reshaping the images to be a single vector
dim(X_train) = c(dim(X_train)[1], prod(dim(X_train)[c(2, 3)]))
dim(X_test) = c(dim(X_test)[1], prod(dim(X_test)[c(2, 3)]))

## Printing the shape of the data sets
print(dim(X_train))
print(dim(X_test))

## Constructing the model
autoencoder <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 784, activation = 'sigmoid')

autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy')

## Fitting the model to the training data
autoencoder %>% fit(X_train, X_train, epochs = 10, batch_size = 32, shuffle = TRUE, 
                    validation_split = 0.2)

## Evaluating model performance
score <- autoencoder %>% evaluate(X_test, X_test)
cat('Test loss:', score["loss"], "\n")

## Predicting on the testing set
preds <- autoencoder %>% predict(X_test)

## Reshaping the testing set and predictions into image format
dim(X_test) = c(dim(X_test)[1], 28, 28)
dim(preds) = c(dim(preds)[1], 28, 28)

## Plotting some results
par(mfcol=c(2,5))
for (i in 11:15){
  img = X_test[i,,]
  img = t(apply(img, 2, rev))
  image(1:28, 1:28, img, xaxt = 'n', yaxt = 'n', 
        xlab = '', ylab = '', main = 'Testing Image')
  
  img = preds[i,,]
  img = t(apply(img, 2, rev))
  image(1:28, 1:28, img, xaxt = 'n', yaxt = 'n',
        xlab = '', ylab = '', main = 'Reconstruction')
}







## -------------------------------------------------------

######################################
## Applying TS Data w/ Autoencoders ##
######################################

library(tsinterp)
library(interpTools)

set.seed(42)

## Simulating time series and scaling values to (0, 1)
N = 100
xt = simXt(N = N, mu = 0, numTrend = 1, a = 1, center = 50,
           numFreq = 2, b = c(0.2, 0.2), w = c(pi/2, pi/4))$Xt
xt = (0.999 - 0.001) * (xt - min(xt)) / (max(xt) - min(xt)) + 0.001

## Plotting the time series
par(mfcol=c(1,1), mar = c(5.1, 4.1, 4.1, 2.1))
plot(xt, type = 'l'); grid()


## Creating K gappy time series
K = 20
xt_gappy = simulateGaps(list(xt), p = 0.05, g = 1, K = K)


## Creating data matrix 
data_gappy = c()
data_target = c()

for (k in 1:K){
  data_gappy = c(data_gappy, xt_gappy[[1]]$p0.05$g1[[k]])
  data_target = c(data_target, xt)
}

data_gappy = ifelse(is.na(data_gappy), 0, data_gappy)
data_gappy = array(matrix(data_gappy, nrow = K, byrow = TRUE), dim = c(K, N))
data_target = array(matrix(data_target, nrow = K, byrow = TRUE), dim = c(K, N))


## Printing the dimensions of the input and target sets
print(dim(data_gappy))
print(dim(data_target))

## Plotting an example
par(mfcol=c(3,1), mar = c(5.1, 4.1, 1, 2.1))
for (i in 1:3){
  plot(data_target[i,], type = 'p', col = ifelse(data_gappy[i,] == 0, 'red', 'black'), 
       lty = 1, lwd = 1, cex = 1, xlab = 'Index', ylab = 'Value')
  lines(data_gappy[i,], type = 'l', col = 'red', lwd = 2)
  grid()
}

array(data_gappy[1,], dim = c(1, length(data_gappy[1,])))


## Constructing and compiling the model
autoencoder <- keras_model_sequential(name = 'Autoencoder') %>%
  layer_masking(mask_value = 0, input_shape = c(N)) %>%
  layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
  layer_dense(units = N, activation = 'sigmoid', name = 'decoder')

autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy')

## Fitting the model to the training data
autoencoder %>% fit(data_gappy, data_target, epochs = 100, batch_size = 32, 
                    shuffle = FALSE, validation_split = 0.2)

## Predicting on the testing set
preds <- autoencoder %>% predict(data_gappy)

dim(preds)

par(mfcol=c(3,1), mar = c(5.1, 4.1, 1, 2.1))
plot(data_target[1,], type = 'l', col = 'black', lty = 1, lwd = 1, cex = 1, 
     xlab = 'Index', ylab = 'Value')
plot(data_gappy[1,], type = 'b', col = 'red', lty = 1, lwd = 1, cex = 1, 
     xlab = 'Index', ylab = 'Value')
plot(preds[1,], type = 'l', col = 'blue', lty = 1, lwd = 1, cex = 1, 
     xlab = 'Index', ylab = 'Value')
grid()


imputed1 = ifelse(data_gappy[1,] == 0, preds[1,], data_target[1, ])
par(mfcol=c(1,1), mar = c(5.1, 4.1, 4.1, 2.1))
plot(data_target[1,], type = 'p', col = ifelse(data_gappy[1,] == 0, 'red', 'black'), lty = 1, lwd = 1, cex = 1, 
     xlab = 'Index', ylab = 'Value')
lines(imputed1, type = 'l', col = 'blue')



dim(preds)
preds[1,]



is.na(data_gappy[1,])




data_gappy

raw_inputs <- list(data_interp[1:10], data_interp[11:20], data_interp[21:30])
padded_inputs <- pad_sequences(raw_inputs, padding = "post"); print(padded_inputs)


embedding <- layer_embedding(input_dim = 5000, output_dim = 16, mask_zero = TRUE)
masked_output <- embedding(padded_inputs);print(masked_output$'_keras_mask')








## LSTM Learning:

library(tensorflow)
library(keras)
library(interpTools)

source('main.R')
source('ititialize.R')
source('estimate.R')
source('simulate.R')









autoencoder_lstm = keras_model_sequential(name = 'Autoencoder LSTM') %>%
  layer_masking(mask_value = -1, input_shape = c(N, 1), name = 'mask') %>%
  layer_lstm(units = 64, name = 'LSTM') %>%
  layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
  layer_dense(units = N, activation = 'sigmoid', name = 'decoder')

summary(autoencoder_lstm)










