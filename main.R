########################################
## Neural Network Time Series Imputer ##
########################################

## MAIN METHOD: 


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)





source('estimate.R')
source('simulate.R')

set.seed(42)
x = simXt(N = 100, mu = 0, numTrend = 1, numFreq = 2)$Xt
x = (x - min(x)) / (max(x) - min(x))
x_0 = simulateGaps(list(x), p = 0.1, g = 1, K = 1)[[1]]$p0.1$g1[[1]]
plot(x, type = 'l', lwd = 1.5); grid()


data = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'noise')
data2 = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'all')
data3 = simulator(x_0, x, n_series = 2, p = 0.1, g = 1, K = 1, random = TRUE, method = 'separate')
lines(data[[2]][1,], type = 'l', col = 'darkorange')
lines(data[[2]][2,], type = 'l', col = 'darkorange')
lines(data2[[2]][1,], type = 'l', col = 'dodgerblue')
lines(data2[[2]][2,], type = 'l', col = 'dodgerblue')
lines(data3[[2]][1,], type = 'l', col = 'red')
lines(data3[[2]][2,], type = 'l', col = 'red')



mt = estimateMt(x, N = length(x), nw = 5, k = 8, pMax = 2)
lines(mt, type = 'l', col = 'red')
detrend = x - mt
plot(detrend, type = 'l')
mean(detrend)
plot(detrend - mean(detrend), type = 'l')

no_mea = detrend - mean(detrend)
mean(no_mea)
