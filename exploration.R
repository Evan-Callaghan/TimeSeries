## Exploration file:

library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)
library(multitaper)
library(stats)

## Simulating a time series
set.seed(42)
x = simXt(N = 500, numTrend = 0, mu = 0, numFreq = 2, b = c(1, 2))$Xt
plot(x, type = 'l', lwd = 2); grid()



plot.frequency.spectrum <- function(X.k, xlimits=c(0,length(X.k))){
  plot.data  <- cbind(0:(length(X.k)-1), Mod(X.k))
  
  # TODO: why this scaling is necessary?
  plot.data[2:length(X.k),2] <- 2*plot.data[2:length(X.k),2] 
  
  plot(plot.data, t="h", lwd=2, main="", 
       xlab="Frequency (Hz)", ylab="Strength", 
       xlim=xlimits, ylim=c(0,max(Mod(plot.data[,2]))))
}


ft = fft(array(x), inverse = FALSE)
plot.frequency.spectrum(ft, xlimits = c(0, length(x)/2))

plot(ft, type = 'l')




## Example:
set.seed(101)
acq.freq <- 200
time <- 1
w <- 2*pi/time
ts <- seq(0, time, 1/acq.freq)
trajectory <- 3 * rnorm(101) + 3 * sin(3 * w * ts)
plot(trajectory, type="l"); grid()

X.k <- fft(trajectory)
plot.frequency.spectrum(X.k, xlimits = c(0, acq.freq/2))

spec.mtm(ts(x), plot = TRUE)
