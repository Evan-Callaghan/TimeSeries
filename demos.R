########################################
## Neural Network Time Series Imputer ##
########################################



## Demonstrations **



## Importing libraries
## -----------------------
library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)


## Defining all functions
## -----------------------


#' initialize
#' 
#' Function to initialize the imputation process. Completes the first step of the designed algorithm 
#' which is to linearly impute the missing values as a starting point.
#' @param x0 {list}; List object containing the original incomplete time series
#' 
initialize_demo <- function(x0){
  gapTrue = ifelse(is.na(x0), NA, TRUE) ## Identifying gap structure
  blocks = tsinterp::findBlocks(gapTrue) ## Computing block structure
  xI = tsinterp::linInt(x0, blocks) ## Initial imputation using linear interpolation
  return(xI)
}


#' estimator
#' 
#' Function to estimate the trend and periodic components of a given time series. Completes the second 
#' step of the designed algorithm.
#' @param x {list}; List object containing a complete time series
#' 
estimator_demo <- function(x, main_demo = FALSE){
  Mt = estimateMt(x = x, N = length(x), nw = 5, k = 8, pMax = 2) ## Estimating trend component
  Tt = estimateTt(x = x - Mt, epsilon = 1e-6, dT = 1, nw = 5, k = 8, sigClip = 0.999) ## Estimating periodic components
  Xt = Mt
  for (i in 1:dim(Tt)[2]){
    Xt = Xt + Tt[,i]
  }
  if (main_demo == TRUE){
    return(Xt)
  }
  else{
    return(list(Mt, Tt, Xt))
    }
}


#' preprocess
#' 
#' Function to preprocess the data before building the neural network imputer. The function performs 0-1
#' standardization for each time series in the input matrix and masks NA values with 0 (necessary for 
#' the neural network).
#' @param x {matrix}; Matrix object containing a time series in each row
#' 
preprocess_demo <- function(x){
  for (i in 1:dim(x)[1]){
    x[i,] = (0.999 - 0.001) * (x[i,] - min(x[i,], na.rm = TRUE)) / 
      (max(x[i,], na.rm = TRUE) - min(x[i,], na.rm = TRUE)) + 0.001 ## Standardizing to the 0-1 scale
    x[i,] = ifelse(is.na(x[i,]), 0, x[i,]) ## Masking NA values
  }
  return(x)
}


#' simulator
#' 
#' Function to simulate a number of time series with on slight perturbations given the estimated
#' structure of the original time series. With each new series, a specified gap structure is 
#' imposed and returned are two data matrices: one with missing gaps (inputs) and one that is
#' complete (targets). Completes the third and fourth steps of the designed algorithm.
#' @param x {list}; List object containing a complete time series 
#' @param n_series {integer}; Number of new time series to construct
#' @param var {float}; Variance for perturbations
#' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#' 
simulator_demo <- function(x, n_series, var, p, g, K){
  N = length(x); M = n_series * K ## Defining useful parameters
  inputs_temp = c(); targets_temp = c() ## Initializing vectors to store values
  
  for (i in 1:n_series){
    x_p = x + rnorm(1, 0, var) + rnorm(N, 0, var) ## Creating small perturbation
    x_g = simulateGaps(list(x_p), p = p, g = g, K = K) ## Imposing gap structure
    
    for (k in 1:K){
      structure = paste0('x_g[[1]]$p', p, '$g', g, '[[k]]')
      inputs_temp = c(inputs_temp, eval(parse(text = structure))) ## Appending inputs
      targets_temp = c(targets_temp, x_p) ## Appending targets
    }
  }
  inputs = array(matrix(inputs_temp, nrow = M, byrow = TRUE), dim = c(M, N)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = M, byrow = TRUE), dim = c(M, N)) ## Properly formatting targets
  inputs = preprocess_demo(inputs) ## Preprocessing inputs
  targets = preprocess_demo(targets) ## Preprocessing targets
  return(list(inputs, targets))
}


#' impute
#' 
#' Function to construct, compile, and fit a neural network model on the simulated training data
#' and then predict on the original time series. Completes the fifth step of the designed algorithm.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param inputs {matrix}; Matrix object containing the input training data
#' @param targets {matrix}; Matrix object containing the target training data
#' 
impute_demo <- function(x0, inputs, targets, return_model = FALSE){
  N = dim(inputs)[2] ## Defining useful parameters
  x0 = as.array(matrix(ifelse(is.na(x0), 0, x0), nrow = 1, byrow = TRUE)) ## Formatting original series
  
  autoencoder = keras_model_sequential(name = 'Autoencoder') %>% ## Constructing the model
    layer_masking(mask_value = 0, input_shape = c(N), name = 'mask') %>%
    layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
    layer_dense(units = N, activation = 'sigmoid', name = 'decoder') 
  
  autoencoder %>% compile( ## Compiling the model
    optimizer = 'adam',
    loss = 'binary_crossentropy')
  
  if (return_model == TRUE){
    return(autoencoder)
  }
  
  autoencoder %>% fit(inputs, targets, epochs = 25, batch_size = 32, ## Fitting the model to the training data
                      shuffle = TRUE, validation_split = 0.2, verbose = 0)
  
  preds = autoencoder %>% predict(x0, verbose = 0) ## Predicting on the original series
  return(preds)
}


## Defining all copied functions from TSinterp
## -----------------------

## Defining function to estimate trend component
estimateMt <- function(x, N, nw, k, pMax) {
  V <- dpss(n = N, nw = 5, k = 8)$v
  test <- dpssap(V, pMax) # fit quadratic
  U <- test[[1]]
  R <- test[[2]]
  Y <- t(V) %*% x
  a <- t(U) %*% Y
  r <- Y - U %*% a
  xhat <- V %*% r + R %*% a
  phat <- R %*% a
  return(phat)
}

## Defining function to estimate periodic component
estimateTt <- function(x, epsilon, dT, nw, k, sigClip, progress=FALSE, freqIn=NULL) {
  
  ################################################################################
  # Algorithm step 1: spectrum/Ftest pilot estimate
  pilot <- spec.mtm(x, deltat=dT, nw=nw, k=k, Ftest=TRUE, plot=FALSE)
  
  if(is.null(freqIn)) {
    ################################################################################
    # Algorithm step 2: estimate significant peaks (sigClip)
    fmesh <- pilot$mtm$Ftest
    fsig <- fmesh > qf(sigClip, 2, pilot$mtm$k)
    floc <- which(fsig==TRUE)
    if(length(floc > 0)) {
      ###########################################################################   
      delta <- floc[2:length(floc)] - floc[1:(length(floc)-1)]
      if(length(which(delta==1)) > 0) {
        bad <- which(delta==1)
        if(!is.null(bad)) {
          if(progress) {
            for(j in 1:length(bad)) {
              cat(paste("Peak at ", formatC(pilot$freq[floc[bad[j]]], width=6, format="f"),
                        "Hz is smeared across more than 1 bin. \n", sep=""))
            } 
          }
        }
        floc <- floc[-bad] # eliminate the duplicates
      }
      
      ################################################################################
      # Algorithm step 3: estimate centers
      dFI <- pilot$freq[2]
      # epsilon <- 1e-10
      maxFFT <- 1e20
      max2 <- log(maxFFT, base=2)
      max3 <- log(maxFFT, base=3)
      max5 <- log(maxFFT, base=5)
      max7 <- log(maxFFT, base=7)
      
      freqFinal <- matrix(data=0, nrow=length(floc), ncol=1)
      
      for(j in 1:length(floc)) {
        if(progress) {
          cat(".")  
        }
        f0 <- pilot$freq[floc[j]]
        
        if(progress) {
          cat(paste("Optimizing Peak Near Frequency ", f0, "\n", sep=""))
        }
        
        # increasing powers of 2,3,5,7 on nFFT until peak estimate converges
        pwrOrig <- floor(log2(pilot$mtm$nFFT)) + 1
        fI <- f0
        converge <- FALSE
        
        for(k7 in 0:max7) {
          for(k5 in 0:max5) {
            for(k3 in 0:max3) {
              for(k2 in 1:max2) {
                
                if(!converge) {
                  nFFT <- 2^pwrOrig * 2^k2 * 3^k3 * 5^k5 * 7^k7
                  tmpSpec <- spec.mtm(x, deltat=dT, nw=5, k=8, plot=FALSE, Ftest=TRUE,
                                      nFFT=nFFT)
                  dF <- tmpSpec$freq[2]
                  f0loc <- which(abs(tmpSpec$freq - f0) <= dF)
                  range <- which(tmpSpec$freq <= (f0+1.1*dFI) & tmpSpec$freq >= (f0-1.1*dFI))
                  
                  fI2 <- tmpSpec$freq[which(tmpSpec$mtm$Ftest == max(tmpSpec$mtm$Ftest[range]))]
                  if(abs(fI - fI2) > epsilon) {
                    fI <- fI2
                  } else {
                    fF <- fI2
                    converge <- TRUE
                  }
                }
              }}}}
        freqFinal[j] <- fF
        if(progress) {
          cat(paste("Final frequency estimate: ", fF, "\n", sep=""))
        }
      }
      if(progress) {
        cat("\n")
      }
    } else {
      freqFinal <- NULL
      floc <- -1
    } # end of "there are freqs detected"
  } else {  # case where frequencies are already obtained
    freqFinal <- freqIn
    floc <- 1:length(freqFinal)
    if(length(freqFinal)==1 & freqFinal[1]==0) {
      floc <- -1
    }
  }
  ################################################################################
  # Algorithm step 4: frequencies obtained, estimate phase and amplitude
  #    by inverting the spectrum (i.e. line component removal)
  if(length(floc) > 1 | floc[1] > 0) {
    sinusoids <- matrix(data=0, nrow=length(x), ncol=length(floc))
    amp <- matrix(data=0, nrow=length(floc), ncol=1)
    phse <- matrix(data=0, nrow=length(floc), ncol=1)
    N <- length(x)
    t <- seq(1, N*dT, dT)
    
    for(j in 1:length(floc)) {
      sinusoids[, j] <- removePeriod(x, freqFinal[j], nw=5, k=8, deltaT=dT, warn=FALSE, prec=1e-10, sigClip=sigClip) 
      fit <- lm(sinusoids[, j] ~ sin(2*pi*freqFinal[j]*t) + cos(2*pi*freqFinal[j]*t) - 1)
      phse[j] <- atan(fit$coef[2] / fit$coef[1])
      amp[j] <- fit$coef[1] / cos(phse[j])
    }
    
    attr(sinusoids, "Phase") <- phse
    attr(sinusoids, "Amplitude") <- amp
    attr(sinusoids, "Frequency") <- freqFinal
    return(sinusoids)
  } else {
    sinusoids <- matrix(data=0, nrow=length(x), ncol=length(floc))
    attr(sinusoids, "Phase") <- 0
    attr(sinusoids, "Amplitude") <- 0
    attr(sinusoids, "Frequency") <- 0
    return(sinusoids)
  }
}

## Defining helper functions:
dpssap <- function(V, maxdeg) {
  
  # Sanity checks
  stopifnot(is.matrix(V), is.numeric(maxdeg), maxdeg>=0)
  N <- length(V[, 1])
  K <- length(V[1, ])
  P <- maxdeg + 1
  timeArr <- 1:N
  
  R <- matrix(data=0, nrow=N, ncol=P)
  U <- matrix(data=0, nrow=K, ncol=P)
  
  # Setup centered time index
  midTime <- (1+N) / 2
  scl <- 2/(N-1)
  timeArrC <- (timeArr - midTime) * scl
  
  # Start with Gegenbauer polynomials; convergence is faster
  alpha <- 0.75
  R[, 1] <- 1.0
  if(maxdeg > 0) {
    R[, 2] <- 2 * alpha * timeArrC
    if(maxdeg > 1) {
      for(j in 2:maxdeg) {
        A1 <- 2 * ( (j-1) + alpha ) / j
        A2 <- ( (j-2) + 2 * alpha ) / j
        
        R[, (j+1)] <- A1 * timeArrC * R[, j] - A2 * R[, (j-1)]
      } # end of loop on higher orders
    } # end of maxdeg > 1
  } # end of maxdeg > 0
  
  # Inner Products of R and V
  for(L in 1:P) {
    Kmin <- ( (L-1) %% 2 ) + 1
    for(k in seq(Kmin, K, 2)) {  # loop on non-zero Slepians
      U[k, L] <- t(V[, k]) %*% R[, L]
    }
  }
  
  # Degree 0, 1 (manual) -- L = degree+1
  for(L in 1:min(2,P)) {
    scl <- 1 / sqrt( sum(U[, L]^2) )
    U[, L] <- U[, L] * scl # orthonormalize
    R[, L] <- R[, L] * scl
  }
  
  # loop on higher degrees, applying Gram-Schmidt only on similar
  # parity functions (as even/odd are already orthogonal in U)
  if( P > 2 ) {
    for(L in 3:P) {
      if(L %% 2 == 0) {
        Kmin <- 2
      } else {
        Kmin <- 1
      }
      for(j in seq(Kmin, L-1, 2)) {
        scl <- sum( U[, L] * U[, j] )
        U[, L] <- U[, L] - scl * U[, j] # Gram-Schmidt
        R[, L] <- R[, L] - scl * R[, j]
      }
      scl <- 1 / sqrt(sum(U[, L]^2))
      U[, L] <- U[, L] * scl  # orthonormalize
      R[, L] <- R[, L] * scl
    }
  }
  
  Hn <- colSums(R^2)
  return(list(U,R,Hn))
}

removePeriod <- function(xd, f0, nw, k, deltaT, warn=FALSE, prec=1e-10, sigClip) {
  
  # xd : data
  # f0 : freq of periodicty to remove
  # nw, k : parameters of multitaper
  # deltaT : parameter of xd
  # prec.st : starting precision for finding a good nFFT for removal
  
  # check to make sure f0 is reasonable, otherwise warn
  N <- length(xd)
  spec.t <- spec.mtm(xd,nw=nw,k=k,Ftest=T,plot=F,nFFT=2^(floor(log(N,2))+2),deltat=deltaT)
  idx <- max(which(spec.t$freq < f0))
  if( max(spec.t$mtm$Ftest[idx],spec.t$mtm$Ftest[idx]) < qf(sigClip,2,(2*k-2)) && warn ) {
    warning("Ftest at frequency f0 not significant. Are you sure you want to remove this?")
  }
  
  # early parameter setup, find a nFFT that gives a bin *very* close to f0, or on top of it
  Nyq <- 1/2/deltaT
  nFFT <- -1
  prec.st <- prec
  while( nFFT < 0 ) {
    nFFT <- findPowers(N,f0,Nyq,prec.st)
    prec.st <- prec.st*10
  }
  
  spec <- spec.mtm(xd,nw=nw,k=k,returnInternals=T,Ftest=T,plot=F,nFFT=nFFT,maxAdaptiveIterations=0,
                   deltat=deltaT)
  
  # parameter setup
  w <- nw/N/deltaT
  df <- 1/nFFT/deltaT
  neh <- max(10,(floor((2*w)/df+1)))
  f0.idx <- seq(along=spec$freq)[spec$freq == (f0 - min(abs(spec$freq - f0))) | spec$freq == (f0 + min(abs(spec$freq - f0)))]
  
  ##########################################################################
  # 
  #  All spectral window work will require the full spectral array
  # 
  ##########################################################################
  # form spectral windows
  dw <- dpss(N,k,5.0)$v*sqrt(deltaT)
  # zero-pad
  dw.z <- rbind(dw,matrix(data=0,nrow=(spec$mtm$nFFT-N),ncol=k))
  # empty window array, nFFT x k
  sw <- matrix(data=0,nrow=spec$mtm$nFFT,ncol=k)
  for(j in 1:k) {
    ft <- fft(dw.z[,j])
    sw[,j] <- c(ft[(spec$mtm$nfreqs+1):spec$mtm$nFFT],ft[1:spec$mtm$nfreqs])
  }
  
  # form estimate of chosen frequency component - takes 0+/- neh from the spectral
  #   window and expands it by multiplying by the CMV at f0
  est <- matrix(data=complex(0,0),nrow=(2*neh+1),ncol=k)
  for(j in 1:k) {
    est[,j] <- spec$mtm$cmv[f0.idx]*(sw[((spec$mtm$nfreqs-1)-neh):((spec$mtm$nfreqs-1)+neh),j])
  }
  
  # subtract from original eigencoefficients
  egn <- spec$mtm$eigenCoefs
  egn <- rbind(Conj(egn[2:spec$mtm$nfreqs,]), egn)
  range <- (f0.idx-neh+spec$mtm$nfreqs) : (f0.idx+neh+spec$mtm$nfreqs)
  if(max(range) > nFFT) {
    # case of folding over the top, i.e. freq too close to Nyq+
    fold <- which(range > nFFT)
    rangeF <- range[fold]
    rangeN <- range[-fold]
    range <- c(1:length(rangeF), rangeN)
  } 
  est2 <- est
  for(j in 1:k) {
    est2[which(range < spec$mtm$nfreqs),j] <- Conj(est2[which(range < spec$mtm$nfreqs),j])
    egn[range,j] <- egn[range,j] - est2[,j]
  }
  
  blank <- matrix(data=0,nrow=nFFT,ncol=1)
  blank[f0.idx] <- spec$mtm$cmv[f0.idx]
  blank[nFFT-f0.idx+2] <- Conj(spec$mtm$cmv[f0.idx])
  inv <- fft(blank,inverse=T)
  inv <- Re(inv)[1:N]
  
  #  cat(paste("Freq: ", spec$freq[f0.idx]," \n",
  #            "Amp : ", sqrt(Mod(spec$mtm$cmv[f0.idx])), "\n",
  #            "Phse: ", Arg(spec$mtm$cmv[f0.idx]), "\n", sep=""))
  return(inv)
}

findPowers <- function(N,f0,Nyq,prec) {
  nFFT <- 1e30
  
  low2 <- 0
  high2 <- floor(log(N,2))+2
  low3 <- 0
  high3 <- floor(log(N,3))+2
  low5 <- 0
  high5 <- floor(log(N,5))+2
  low7 <- 0
  high7 <- floor(log(N,7))+2
  for(i in low2:high2) {
    for(j in low3:high3) {
      for(k in low5:high5) {
        for(l in low7:high7) {
          att <- 2^i * 3^j * 5^k * 7^l
          if((att > 2*N) & att < 100*N) {
            df <- (Nyq*2)/att
            if( abs(trunc(f0/df)*df - f0) < prec ) {
              if(att < nFFT) {
                nFFT <- att 
              }
            }
          } # big enough
        } # end of 7
      } # end of 5
    } # end of 3
  } # end of 2
  if(nFFT == 1e30) {
    return(-1)
  } else {
    return(nFFT)
  }
}


## Defining main method
## -----------------------


#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param n_series {integer}; Number of new time series to construct
#' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#' @param var {float}; Variance for perturbations
#'
main <- function(x0, max_iter, n_series, p, g, K, var){
  
  results = matrix(NA, ncol = length(x0), nrow = max_iter) ## Defining matrix to store results
  xI = initialize(x0) ## Step 1
  
  for (i in 1:max_iter){
    x_est = estimator(xI) ## Step 2
    data = simulator(x_est, n_series, var, p, g, K); inputs = data[[1]]; targets = data[[2]] ## Steps 3/4
    pred = impute(x0, inputs, targets) ## Step 5
    xI = ifelse(is.na(x0), pred, x0); results[i,] = xI ## Step 6
    print(paste0('Interation ', i))
  }
  return(colMeans(results))
}









## Defining the time series for demo
set.seed(42)
par(mfrow = c(1,1))
xt = simXt(N = 100, mu = 0, numTrend = 1, numFreq = 2)$Xt
xt = (xt - min(xt)) / (max(xt) - min(xt))
x_gapped = simulateGaps(list(xt), p = 0.1, g = 2, K = 1)
x_gappy = x_gapped[[1]]$p0.1$g2[[1]]
plot(xt, type = 'l', main = 'Demo Time Series'); grid()
lines(xt, type = 'p', col = ifelse(is.na(x_gappy), 'dodgerblue', 'black'), cex = 0.8)
legend('topleft', legend = c('Observed', 'Missing'), col = c('black', 'dodgerblue'), pch = 1)


## -----------------------
## Initialization (Step 1)
step_1_demo <- function(x0){
  xI = initialize_demo(x0)
  plot(xI, type = 'l', main = 'Demo Time Series', ylab = 'xt', col = 'red'); grid()
  lines(xt, type = 'l')
  lines(xt, type = 'p', col = ifelse(is.na(x_gappy), 'dodgerblue', 'black'), cex = 0.8)
  legend('topleft', legend = c('Observed', 'Missing', 'Lin. Imputation'),  lty = c(0, 0, 1), 
         col = c('black', 'dodgerblue', 'red'), pch = c(1, 1, -1))
  return(xI)
}
step1 = step_1_demo(x_gappy)


## -----------------------
## Trend and Periodic Estimation (Step 2)
step_2_demo <- function(xI){
  x_est = estimator_demo(xI)
  Mt = x_est[[1]]; Tt = x_est[[2]]; Xt = x_est[[3]]
  plot(xI, type = 'l', main = 'Demo Time Series', ylab = 'xt'); grid()
  lines(Mt, type = 'l', col = 'dodgerblue')
  lines(Xt, type = 'l', col = 'red')
  legend('topleft', legend = c('Step 1 Output', 'Trend', 'Trend + Periodic'), 
         col = c('black', 'dodgerblue', 'red'), lty = 1, lwd = 2)
  return(Xt)
}
step2 = step_2_demo(step1)


## -----------------------
## Data Simulation (Step 3/4)
step_34_demo <- function(x_est, n_series, var, p, g, K){
  x = simulator_demo(x_est, n_series, var, p, g, K)
  inputs = x[[1]]; targets = x[[2]]
  par(mfcol=c(3,1), mar = c(5.1, 4.1, 1, 2.1))
  x_est_scaled = (x_est - min(x_est)) / (max(x_est) - min(x_est))
  for (i in 1:3){
    plot(targets[i,], type = 'l', col = 'dodgerblue', lwd = 2, ylab = 'xt', 
         main = paste0('Simulated Time Series ', i, ' (K = 1)')); 
    lines(x_est_scaled, type = 'l', col = 'black'); grid()
    lines(targets[i,], type = 'p', cex = 1.5,
          col = ifelse(inputs[i,] == 0, 'red', adjustcolor( "red", 0.01)))
    legend('topleft', legend = c('Estimated TS', 'Perturbed TS', 'Missing Points'), 
           pch = c(-1, -1, 1), lty = c(1, 1, 0), col = c('black', 'dodgerblue', 'red'))
  }
  return(x)
}
step34 = step_34_demo(step2, n_series = 3, var = 0.05, p = 0.1, g = 2, K = 1)
inputs = step3[[1]]; targets = step3[[2]]


## -----------------------
## Building the Neural Network (Step 5)
step_5_demo <- function(x0, inputs, targets, return_model){
  model = impute_demo(x0, inputs, targets, return_model)
  print(model)
}
step_5_demo(x_gappy, inputs, targets, return_model = TRUE)


## -----------------------
## Full simulation
main_demo <- function(x0, max_iter, n_series, p, g, K, var){
  
  results = matrix(NA, ncol = length(x0), nrow = max_iter) ## Defining matrix to store results
  xI = initialize_demo(x0) ## Step 1
  
  for (i in 1:max_iter){
    x_est = estimator_demo(xI, main_demo = TRUE) ## Step 2
    data = simulator_demo(x_est, n_series, var, p, g, K); inputs = data[[1]]; targets = data[[2]] ## Steps 3/4
    pred = impute_demo(x0, inputs, targets) ## Step 5
    xI = ifelse(is.na(x0), pred, x0); results[i,] = xI ## Step 6
    print(paste0('Interation ', i))
  }
  return(colMeans(results))
}
nn_predictions = main_demo(x_gappy, max_iter = 10, n_series = 40, p = 0.1, g = 2, K = 5, var = 0.05)


par(mfrow = c(1,1))
plot(xt, type = 'l', lwd = 2); grid()
lines(nn_predictions, type = 'l', col = 'red')
lines(xt, type = 'p', col = ifelse(is.na(x_gappy), 'dodgerblue', adjustcolor( "red", 0.01)))
legend('topleft', legend = c('True TS', 'Imputed TS', 'Missing Point'), 
       col = c('black', 'red', 'dodgerblue'), lty = c(1, 1, 0), pch = c(-1, -1, 1))


eval_performance(x = xt, X = nn_predictions, gappyx = x_gappy)$RMSE



x_hwi = parInterpolate(x_gapped, methods = 'HWI')[[1]]$HWI$p0.1$g2[[1]]
eval_performance(x = xt, X = x_hwi, gappyx = x_gapped[[1]]$p0.1$g2[[1]])$RMSE




















## Comparing performance across methods:
x = simXt(N = 500, numTrend = 1, a = 10)$Xt; x = (x - min(x)) / (max(x) - min(x))
x_gappy = simulateGaps(list(x), p = 0.1, g = 1, K = 1)
plot(x, type = 'l')
lines(x_gappy[[1]]$p0.1$g1[[1]], type = 'l', col = 'cyan')
grid()


interp = parInterpolate(x_gappy, methods = c('HWI', 'NN', 'LOCF', 'LWMA'))
x_hwi = interp[[1]]$HWI$p0.1$g1[[1]]
x_nn = interp[[1]]$NN$p0.1$g1[[1]]
x_locf = interp[[1]]$LOCF$p0.1$g1[[1]]
x_lwma = interp[[1]]$LWMA$p0.1$g1[[1]]

x_neural = main(x_gappy[[1]]$p0.1$g1[[1]], max_iter=10, n_series=50, p=0.1, g=1, K=10, var=0.03)


eval_performance(x = x, X = x_hwi, gappyx = x_gappy[[1]]$p0.1$g1[[1]])$RMSE
eval_performance(x = x, X = x_nn, gappyx = x_gappy[[1]]$p0.1$g1[[1]])$RMSE
eval_performance(x = x, X = x_locf, gappyx = x_gappy[[1]]$p0.1$g1[[1]])$RMSE
eval_performance(x = x, X = x_lwma, gappyx = x_gappy[[1]]$p0.1$g1[[1]])$RMSE
eval_performance(x = x, X = x_neural, gappyx = x_gappy[[1]]$p0.1$g1[[1]])$RMSE










