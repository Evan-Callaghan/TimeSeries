########################################
## Neural Network Time Series Imputer ##
########################################



#' Algorithm
#' 
#' 1. Interpolate missing values in the original time series using linear interpolation.
#' 
#' 2. a. Estimate the trend components of the input time series
#'    b. Estimate the periodic component(s) of the input time series
#'    
#' 3. Simulate a sufficient number of time series with the same trend and periodic structure 
#'    using small perturbations (inputs). 
#'    
#' 4. Impose a gap structure on the simulated time series (i.e., the same gap structure as 
#'    observed in the original time series) and produce K unique ‘gappy’ time series (targets).
#'    
#' 5. Construct, compile, and fit a neural network autoencoder model using the newly created input 
#'    and target time series. Using the autoencoder, predict on the original time series.
#'    
#' 6. Extract the predicted values of the missing data points to complete the original time series. 
#' 
#' 7. Repeat steps 2-6 using the output of Step 6 as the input for Step 2, each time storing the 
#'    predictions.
#'    
#' 8. Return the final predictions as the average predictions over the total number of iterations. 
#' 


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
initialize <- function(x0){
  gapTrue = ifelse(is.na(x0), NA, TRUE) ## Identifying gap structure
  blocks = tsinterp::findBlocks(gapTrue) ## Computing block structure
  xI = tsinterp::linInt(x0, blocks) ## Initial imputation using linear interpolation
  return(xI)
}


#' estimator
#' 
#' Function to estimate the trend and periodic components of a given time series. Completes the second 
#' step of the designed algorithm. Returns the estimated trend + periodic series.
#' @param x {list}; List object containing a complete time series
#' 
estimator <- function(x){
  Mt = estimateMt(x = x, N = length(x), nw = 5, k = 8, pMax = 2) ## Estimating trend component
  Tt = estimateTt(x = x - Mt, epsilon = 1e-6, dT = 1, nw = 5, k = 8, sigClip = 0.999) ## Estimating periodic components
  Xt = Mt
  for (i in 1:dim(Tt)[2]){
    Xt = Xt + Tt[,i]
  }
  return(Xt)
}


#' preprocess
#' 
#' Function to preprocess the data before building the neural network imputer. The function performs 0-1
#' standardization for each time series in the input and target matrix.
#' @param inputs {matrix}; Matrix object containing the input training data (i.e., an incomplete 
#'                         time series in each row)
#' @param targets {matrix}; Matrix object containing the target training data (i.e., a complete 
#'                         time series in each row)
#' 
preprocess <- function(inputs, targets){
  for (i in 1:dim(inputs)[1]){
    inputs[i,] = (inputs[i,] - min(targets[i,])) / (max(targets[i,]) - min(targets[i,])) 
    targets[i,] = (targets[i,] - min(targets[i,])) / (max(targets[i,]) - min(targets[i,]))
    
    inputs[i,] = ifelse(is.na(inputs[i,]), -1, inputs[i,])
  }
  return(list(inputs, targets))
}


#' simulator
#' 
#' Function to simulate a number of time series with on slight perturbations given the estimated
#' structure of the original time series. With each new series, a specified gap structure is 
#' imposed and returned are two data matrices: one with missing gaps (inputs) and one that is
#' complete (targets). Completes the third and fourth steps of the designed algorithm.
#' @param x0 {list}; List object containing a the original incomplete time series 
#' @param xI {list}; List object containing a complete time series (i.e., up-to-date imputation)
#' @param x {list}; List object containing a complete time series (i.e., trend + period estimation)
#' @param n_series {integer}; Number of new time series to construct
#' @param p {flaot}; Proportion of missing data to be applied to the simulated series
#' @param g {integer}; Gap width of missing data to be applied to the simulated series
#' @param K {integer}; Number of output series with a unique gap structure for each simulated series
#' @param random {boolean}; Indicator of whether or not the gap placement is randomized
#' 
simulator <- function(x0, xI, x, n_series, p, g, K, random = TRUE){
  if (random == FALSE){K = 1}
  N = length(x); M = n_series * K ## Defining useful parameters
  inputs_temp = c(); targets_temp = c() ## Initializing vectors to store values
  w = xI - x ## Computing the residual noise
  w = fft(w, inverse = FALSE) ## Converting noise to frequency domain
  
  if (random == TRUE){
    for (i in 1:n_series){
      w_p = w * complex(modulus = 1, argument = runif(N, 0, 2*pi)) ## Creating small perturbation
      w_t = as.numeric(fft(w_p, inverse = TRUE)) / N ## Converting back to time domain
      x_p = x + w_t ## Adding perturbed noise back to trend and periodic
      x_g = simulateGaps(list(x_p), p = p, g = g, K = K) ## Imposing gap structure
      
      for (k in 1:K){
        structure = paste0('x_g[[1]]$p', p, '$g', g, '[[k]]')
        inputs_temp = c(inputs_temp, eval(parse(text = structure))) ## Appending inputs
        targets_temp = c(targets_temp, x_p) ## Appending targets
      }
    }
  }
  
  else if (random == FALSE){
    g_index = which(is.na(x0)) ## Defining useful parameters
    for (i in 1:n_series){
      w_p = w * complex(modulus = 1, argument = runif(100, 0, 2*pi)) ## Creating small perturbation
      w_t = as.numeric(fft(w_p, inverse = TRUE)) / N ## Converting back to time domain
      x_p = x + w_t ## Adding perturbed noise back to trend and periodic
      x_g = x_p; x_g[g_index] = NA ## Imposing non-randomized gap structure
      inputs_temp = c(inputs_temp, x_g) ## Appending inputs
      targets_temp = c(targets_temp, x_p) ## Appending targets
    }
  }
  
  inputs = array(matrix(inputs_temp, nrow = M, byrow = TRUE), dim = c(M, N)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = M, byrow = TRUE), dim = c(M, N)) ## Properly formatting targets
  preprocessed = preprocess(inputs, targets) ## Preprocessing
  inputs = preprocessed[[1]]; targets = preprocessed[[2]]
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
impute <- function(x0, inputs, targets){
  N = dim(inputs)[2] ## Defining useful parameters
  x0 = as.array(matrix(ifelse(is.na(x0), -1, x0), nrow = 1, byrow = TRUE)) ## Formatting original series
  
  inputs = tf$constant(inputs) ## Creating input tensors
  targets = tf$constant(targets) ## Creating target tensors
  x0 = tf$constant(x0) ## Creating prediction tensor
  
  autoencoder = keras_model_sequential(name = 'Autoencoder') %>% ## Constructing the model
    layer_masking(mask_value = -1, input_shape = c(N)) %>%
    layer_dense(units = 32, activation = 'relu', name = 'encoder') %>%
    layer_dense(units = N, activation = 'sigmoid', name = 'decoder') 
  
  autoencoder %>% compile( ## Compiling the model
    optimizer = 'adam', loss = 'binary_crossentropy')
  
  autoencoder %>% fit(inputs, targets, epochs = 25, batch_size = 32, ## Fitting the model to the training data
                      shuffle = TRUE, validation_split = 0.2, verbose = 0)
  
  preds = autoencoder %>% predict(x0, verbose = 0) ## Predicting on the original series
  return(preds)
}


## Defining all copied functions from TSinterp
## -----------------------
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


## Defining helper functions
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
#'
main <- function(x0, max_iter, n_series, p, g, K){
  
  # Defining matrix to store results
  results = matrix(NA, ncol = length(x0), nrow = max_iter)
  
  ## Step 1: Linear imputation
  xI = initialize(x0)
  
  for (i in 1:max_iter){
    ## Step 2: Getting an estimate for trend + period
    x_estimate = estimator(xI)
    
    ## Step 3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xI, x_estimate, n_series, p, g, K, random = TRUE)
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    pred = impute(x0, inputs, targets)
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xI = ifelse(is.na(x0), pred, x0); results[i,] = xI
    print(paste0('Iteration ', i))
  }
  return(colMeans(results))
}



## -----------------------
## Continuation of demo:


## Generating a time series
set.seed(42)
x = simXt(N = 100, mu = 0, numTrend = 1, numFreq = 2)$Xt
x = (x - min(x)) / (max(x) - min(x))
plot(x, type = 'l', lwd = 1.5); grid()

## Imposing a gap structure (p = 10% and g = 1)
x_0 = simulateGaps(list(x), p = 0.1, g = 1, K = 1)[[1]]$p0.1$g1[[1]]

## Calling the Neural Network Imputer
x_I = main(x_0, max_iter = 10, n_series = 100, p = 0.1, g = 1, K = 5)

## Plotting the imputation vs. ground truth
plot(x_I, type = 'l', col = 'red', main = 'Imputation Results', xlab = 'Time', ylab = 'X')
lines(which(is.na(x_0)), x_I[which(is.na(x_0))], type = 'p', col = 'black', 
      pch = 21, bg = 'red', cex = 0.7)
lines(x, type = 'l', lwd = 2); grid()
legend('topleft', legend = c('Original TS', 'Imputed Value'), 
       lty = c(1, NA), pch = c(NA, 16), cex = 1, lwd = c(2, 2),
       col = c('black', 'red'))

















x_i = initialize(x_0)
x_est = estimator(x_i)
lines(x_est, type = 'l', col = 'red', lwd = 0.5)

plot(x-x_est, type = 'l', lwd = 1.5); grid()
mean(x-x_est)


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










