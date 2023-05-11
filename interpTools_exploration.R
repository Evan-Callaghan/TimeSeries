#install.packages("remotes")
#remotes::install_github("wesleyburr/tsinterp")
#remotes::install_github("castels/interpTools")

## Importing libraries
library(remotes)
library(devtools)
library(tsinterp)
library(interpTools)

## Generating the time series
xt = simXt(N = 1000, mu = 0, numTrend = 1, numFreq = 2)$Xt
plot(xt, type = 'l')
grid()

## Making the time-series into list type
xt = list(xt)

## Created gappy data-frame
xt_gappy = simulateGaps(xt, p = 0.1, g = 5, K = 1)
plot(unlist(xt), type = 'l', col = 'red', lty = 1, lwd = 0.5)
lines(xt_gappy[[1]]$p0.1$g5[[1]], type = 'l', col = 'black', lwd = 2)
grid()

## Interpolating
interp_ts = parInterpolate(xt_gappy, methods = c('LOCF', 'NN', 'HWI'))[[1]]

## Extracting interpolated time-series
xt_locf = interp_ts$LOCF$p0.1$g5[[1]]
xt_nn = interp_ts$NN$p0.1$g5[[1]]
xt_hwi = interp_ts$HWI$p0.1$g5[[1]]

## Plotting the LOCF interpolated time-series
plot(unlist(xt), type = 'l', col = 'red', lty = 1, lwd = 0.5)
lines(xt_gappy[[1]]$p0.1$g5[[1]], type = 'l', col = 'black', lwd = 2)
lines(xt_locf, type = 'l', col = 'blue', lwd = 0.5)
grid()

## Plotting the NN interpolated time-series
plot(unlist(xt), type = 'l', col = 'red', lty = 1, lwd = 0.5)
lines(xt_gappy[[1]]$p0.1$g5[[1]], type = 'l', col = 'black', lwd = 2)
lines(xt_nn, type = 'l', col = 'blue', lwd = 0.5)
grid()

## Plotting the HWI interpolated time-series
plot(unlist(xt), type = 'l', col = 'red', lty = 1, lwd = 0.5)
lines(xt_gappy[[1]]$p0.1$g5[[1]], type = 'l', col = 'black', lwd = 2)
lines(xt_hwi, type = 'l', col = 'blue', lwd = 0.5)
grid()

## Computing the performance
eval_performance(x = xt[[1]], X = xt_locf, gappyx = xt_gappy[[1]]$p0.1$g5[[1]])$RMSE
eval_performance(x = xt[[1]], X = xt_nn, gappyx = xt_gappy[[1]]$p0.1$g5[[1]])$RMSE
eval_performance(x = xt[[1]], X = xt_hwi, gappyx = xt_gappy[[1]]$p0.1$g5[[1]])$RMSE





library(itsmr)

plot(unlist(xt), type = 'l', col = 'red', lty = 1, lwd = 0.5)
plota(unlist(xt))