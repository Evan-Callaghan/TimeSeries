## Exploration file:

library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)
library(multitaper)
library(stats)

## Simulating a time series
set.seed(12)
idx <- 0:63
x = simXt(N = 64, numTrend = 0, mu = 0, numFreq = 2)$Tt
plot(idx, x, type = 'b', lwd = 2); grid()


ft = fft(array(x), inverse = FALSE)
plot(idx, Mod(ft), type = 'h', lwd = 4, xlim = c(0,length(x)/2)); grid()


x_inv = fft(ft, inverse = TRUE)
plot(idx, x_inv, type = 'b', lwd = 2); grid()


## Trying with noise
w = rnorm(100,0,0.5) + sin(0:99); plot(0:99, w, type = 'l', lwd = 2); grid()
f_w = fft(array(w), inverse = FALSE)
plot(0:99, Mod(f_w)/length(w), type = 'h', lwd = 4); grid()


spec.mtm(ts(x), plot = TRUE)
spec <- spec.mtm(x, nw=5.0, k=8, plot=FALSE, deltat=1)
acv <- SpecToACV(spec,maxlag=length(x))
acv
plot(acv, type = 'l')






SpecToACV <- function(spec,maxlag){
  s <- spec$spec
  dF <- spec$freq[2] 
  x <- matrix(data=0,nrow=(spec$mtm$nfreqs-1)*2,ncol=1)
  x[1:spec$mtm$nfreqs] = s*dF
  x[(spec$mtm$nfreqs+1):length(x)] <- x[(spec$mtm$nfreqs-1):2]
  x <- as.complex(x)
  x <- Re(fft(x,inverse=TRUE))
  x[1:(maxlag+1)]
}







plot.frequency.spectrum <- function(X.k, xlimits=c(0,length(X.k))){
  plot.data  <- cbind(0:(length(X.k)-1), Mod(X.k))
  
  # TODO: why this scaling is necessary?
  plot.data[2:length(X.k),2] <- 2*plot.data[2:length(X.k),2] 
  
  plot(plot.data, t="h", lwd=2, main="", 
       xlab="Frequency (Hz)", ylab="Strength", 
       xlim=xlimits, ylim=c(0,max(Mod(plot.data[,2]))))
}
plot.frequency.spectrum(ft)










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





## Exploration:
t <- 0:63
s <- sin(2*pi*t/16)   # Frequency = 1/16
plot(t,s,xlim=c(0,36), type = 'b', lwd = 2) 
title(main="Sine Wave, frequency=1/16")
abline(h = 0); grid()

## Plotting FFT of s
idx <- 0:63
f_s <- fft(s)
plot(idx, abs(f_s), type = 'h', lwd = 4)
title("FFT of sine wave"); grid()

## Creating cosine wave with same frequency
c <- cos(2*pi*t/16) #cosine,Frequency = 1/16
matplot(t,cbind(s,c), type = "b",pch = 1:2,col = 1:2, lwd = 2)
legend(x='topright',pch=1:2,legend=c("sine","cosine"),col=1:2)
grid()

## Comparing fft values at same index
f_c <- fft(c)
mat <- cbind(f_s,f_c)
print(mat[5,],digits=3)

## Creating sine wave with frequency of 1/8
n = sin(2*pi*t/8)  # freq=1/8
signal = s + n
plot(t,signal,type="b", lwd = 2)
title("Time series of noisy signal"); grid()

## Plotting FFT of two signals combined
f_signal = fft(signal)
plot(idx,abs(f_signal), type = 'h', lwd = 4)
title("FFT Of Signal & Noise")

## Applying low pass filter to remove the higher frequency components
f_filtered <- f_signal
f_filtered[1+6:(64-6)] = 0
plot(idx, abs(f_filtered), type = 'h', lwd = 4)

## Applying inverse FFT
filtered <- fft(f_filtered,inverse=TRUE)/64
plot(t, Re(filtered), type = "b")
title("Time series after low-pass filter"); grid()


## -------------------------------
srate = 1000
time = seq(0, 1, 1/srate)
freq = 6
i = complex(real = 0, imaginary = 1)

realval_sine = sin(2*pi*freq*time)
complex_sine = exp(i*2*pi*freq*time)

par(mfrow = c(3,1))
plot(time, realval_sine, type = 'l', lwd = 2, main = 'Real-Valued Sine'); grid()
plot(time, Im(complex_sine), type = 'l', lwd = 2, col = 'blue',  main = 'Imaginary (sine) Component of Complex Sine'); grid()
plot(time, Re(complex_sine), type = 'l', lwd = 2, col = 'red', main = 'Real (cosine) Component of Complex Sine'); grid()



sine1 = 3 * cos(2*pi*freq*time + 0)
sine2 = 2 * cos(2*pi*freq*time + pi/6)
sine3 = 1 * cos(2*pi*freq*time + pi/3)

fCoefs1 = fft(sine1) / length(time)
fCoefs2 = fft(sine2) / length(time)
fCoefs3 = fft(sine3) / length(time)

hz = seq(0, srate/2, 1/(floor(length(time)/2) + 1))



plot(time, sine1, type = 'l')
sine1_fft = fft(sine1)
plot(time, abs(sine1_fft) / length(time), type = 'l', xlim = c(0, 0.02))

1i


## ---------------------------

# If y <- fft(z), then z is fft(y, inverse = TRUE) / length(y)

x <- 1:4
fft(fft(x), inverse = TRUE) / length(x)
all(fft(fft(x), inverse = TRUE)/(x*length(x)) == 1+0i)

eps <- 1e-11 ## In general, not exactly, but still:
for(N in 1:130) {
  cat("N=",formatC(N,wid=3),": ")
  x <- rnorm(N)
  if(N %% 5 == 0) {
    m5 <- matrix(x,ncol=5)
    cat("mvfft:",all(apply(m5,2,fft) == mvfft(m5)),"")
  }
  dd <- Mod(1 - (f2 <- fft(fft(x), inverse=TRUE)/(x*length(x))))
  cat(if(all(dd < eps))paste(" all < ", formatC(eps)) else
    paste("NO: range=",paste(formatC(range(dd)),collapse=",")),"\n")
}

plot(fft(c(9:0,0:13, numeric(301))), type = "l")
periodogram <- function(x, mean.x = mean(x)) { # simple periodogram
  n <- length(x)
  x <- unclass(x) - mean.x
  Mod(fft(x))[2:(n%/%2 + 1)]^2 / (2*pi*n) # drop I(0)
}
data(sunspots)
plot(10*log10(periodogram(sunspots)), type = "b", col = "blue")



