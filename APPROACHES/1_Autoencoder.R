############################################
## Approach 1: Neural Network Autoencoder ##
############################################


## Importing libraries
## -----------------------

library(dplyr)
library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)


## Configuring set-up (not always necessary)
## -----------------------

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}


## Defining all functions
## -----------------------


#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' @param Architecture {integer}; Desired neural network architecture (numerically encoded)
#'
main <- function(x0, max_iter, p, g, train_size, method = 'all', Mod, Arg){
  
  ## Defining useful variables
  N = length(x0)
  
  ## Defining matrix to store results
  results = matrix(NA, ncol = N, nrow = max_iter)
  
  ## Building the neural network model
  model = get_model(N)
  
  ## Step 1: Linear imputation
  xV = initialize(x0)
  
  for (i in 1:max_iter){
    
    ## Step 2: Detecting trend and periodic components
    components = estimator(xV)
    Mt = components[[1]]; Tt = components[[2]]; Xt = components[[3]]
    
    ## Steps 3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xV, Mt, Xt, p, g, train_size, method, Mod, Arg)
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    pred = imputer(x0, inputs, targets, model)
    
    ## Adjusting 'pred' to include trend and mean
    pred_adj = pred[1,] 
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), pred, x0); results[i,] = xV
    #print(paste0('Iteration ', i))
  }
  return(colMeans(results))
}


#' initialize
#' 
#' Function to initialize the imputation process. Completes the first step of the designed algorithm 
#' which is to linearly impute the missing values as a starting point.
#' @param x0 {list}; List object containing the original incomplete time series
#' 
initialize <- function(x0){
  gapTrue = ifelse(is.na(x0), NA, TRUE) ## Identifying gap structure
  blocks = findBlocks(gapTrue) ## Computing block structure
  xV = linInt(x0, blocks) ## Initial imputation using linear interpolation
  return(xV)
}


#' findBlocks
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Computes the
#' missing data structure and returns info regarding where the missing values are located within
#' the time series. 
#' @param mask {list}; List object containing TRUE where there is a value and NA when missing
#' 
findBlocks <- function(mask) {
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
  blocks
}


#' linInt
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Takes the 
#' original incomplete time series and performs linear imputation. 
#' @param dat {list}; List object containing the original incomplete time series
#' @param mask {list}; List object containing TRUE where there is a value and NA when missing
#' 
linInt <- function(dat, blocks) {
  nGap <- length(blocks[,1])
  for(j in 1:nGap) {
    dY <- (dat[blocks[j,2]+1] - dat[blocks[j,1]-1])/(blocks[j,3]+1)
    st <- dat[blocks[j,1]-1]
    lt <- dat[blocks[j,2]+1]
    if(dY != 0) {
      fill <- seq(st,lt,dY)
    } else {
      fill <- rep(st,blocks[j,2]-blocks[j,1]+3) 
    }
    fill <- fill[c(-1,-length(fill))]
    dat[blocks[j,1]:blocks[j,2]] <- fill
  }
  dat
}


#' estimator
#' 
#' Main function for the "estimate" file. The "method" parameter determines which method should 
#' be returned:
#' Case 1: Returning the trend + periodic component(s)
#' Case 2: Returning the trend component
#' Case 3: Returning the periodic component(s)
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param method {string}; Case in c('Xt', 'Mt', 'Tt')
#' 
estimator <- function(xV){
  Mt = estimateMt(x = xV, N = length(xV), nw = 5, k = 8, pMax = 2) ## Estimating trend component
  Tt = estimateTt(x = xV - Mt, epsilon = 1e-6, dT = 1, nw = 5, k = 8, sigClip = 0.999) ## Estimating periodic components
  Xt = Mt ## Combining trend and periodic
  for (i in 1:dim(Tt)[2]){
    Xt = Xt + Tt[,i]
  }
  # if (method == 'Xt'){return(Xt)}
  # else if (method == 'Mt'){return(Mt)}
  # else if (method == 'Tt'){return(Tt)}
  return(list(Mt, Tt, Xt))
}


#' estimateMt
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Takes a 
#' complete time series and other parameters to estimate trend component of a series. 
#' @param x {list}; List object containing a complete time series
#' @param N {?}; 
#' @param nw {?}; 
#' @param k {?}; 
#' @param pMax {integer}; 
#' 
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


#' estimateTt
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Takes a 
#' complete time series and other parameters to estimate the periodic component(s) of a series. 
#' @param x {list}; List object containing a complete time series
#' @param epsilon {?}; 
#' @param dT {integer}; Change in time between data points in series
#' @param nw {?}; 
#' @param sigClip {?}; 
#' @param progress {boolean}; 
#' @param freqIn {?};
#' 
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


#' dpssap
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Not entirely
#' sure what it does. Something to find a trend component.
#' @param V {?}; 
#' @param maxdeg {integer}; Maximum degree polynomial to fit as trend component
#' 
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


#' removePeriod
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Removes the 
#' periodic component(s) from the input series. 
#' @param xd {list}; List object containing a complete time series
#' @param f0 {float}; Frequency of periodicity to remove
#' @param nw {?}; 
#' @param k {?}; 
#' @param deltaT {integer}; Change in time between data points in series
#' @param warn {boolean}; 
#' @param prec {float}; Starting precision for finding a good nFFT for removal
#' @param sigClip {?};
#' 
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


#' findPowers
#' 
#' Function taken directly from Wesley's implementation of the 'TSinterp' package. Not sure what 
#' this function does.
#' @param N {?}; 
#' @param f0 {float}; Frequency of periodicity to remove
#' @param Nyq {?}; 
#' @param prec {float}; Starting precision for finding a good nFFT for removal
#' 
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


#' simulator
#' 
#' Main function for the "simulate" file. The "method" parameter determines which method is used
#' to generate the neural network training data:
#' 
#' Method 1: Data is generated by first estimating the trend and periodic components, computing the 
#' residual noise between the estimate and the current version of the imputed series, perturbing the 
#' noise series, and then adding the perturbed noise back to the estimate series.
#' 
#' Method 2: Data is generated by first removing the trend and the mean, perturbing the de-trended
#' series, and then adding the mean and trend back to the perturbed series.
#' 
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param train_size {integer}; Number of new time series to construct
#' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' 
simulator <- function(x0, xV,  Mt, Xt, p, g, train_size, method = 'noise', Mod, Arg){
  
  ## Case 1: 
  if (method == 'noise'){
    data = simulator_noise(x0, xV, Xt, p, g, train_size, Mod, Arg) ## Simulating data
    return(data)
  }
  
  ## Case 2: 
  else if (method == 'all'){
    data = simulator_all(x0, xV, Mt, p, g, train_size, Mod, Arg) ## Simulating data
    return(data)
  }
}


#' simulator_noise
#' 
#' Function to simulate training data for method = 'noise' (i.e., follows Method 1 described above).
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param xE {list}; List containing the current trend + periodic estimates of xV ("x estimate")
#' @param train_size {integer}; Number of new time series to construct
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' 
simulator_noise <- function(x0, xV, xE, p, g, train_size, Mod, Arg){
  
  ## Defining useful parameters
  N = length(xV) 
  #x_g_text = paste0('interpTools::simulateGaps(list(x_p), p, g, K = 1)[[1]]$p', p, '$g', g, '[[1]]')
  
  inputs_temp = c(); targets_temp = c() ## Initializing vectors to store values
  w = xV - xE ## Computing the residual noise
  w_f = fft(w, inverse = FALSE) ## Converting noise to frequency domain
  
  for (i in 1:train_size){
    set.seed(i) ## Setting a common seed
    #w_p = w_f * complex(modulus = runif(N, 1-Mod, 1+Mod), argument = runif(N, 0-Arg, 0+Arg)) ## Creating small perturbation
    #w_t = as.numeric(fft(w_p, inverse = TRUE)) / N ## Converting back to time domain
    #x_p = xE + w_t ## Adding perturbed noise back to trend and periodic
    x_p = xV
    x_g = create_gaps(x_p, x0, p, g); x_g[which(is.na(x_g))] = 0 ## Imposing randomized gap structure
    #x_g = eval(parse(text = x_g_text));
    
    inputs_temp = c(inputs_temp, x_g) ## Appending inputs
    targets_temp = c(targets_temp, x_p) ## Appending targets
  }
  
  inputs = array(matrix(inputs_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting targets
  return(list(inputs, targets))
}


#' simulator_all
#' 
#' Function to simulate training data for method = 'all' (i.e., follows Method 2 described above).
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param Mt {list}; List containing the estimated trend component
#' @param train_size {integer}; Number of new time series to construct
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#' 
simulator_all <- function(x0, xV, Mt, train_size, Mod, Arg){
  N = length(xV) ## Defining useful parameters
  inputs_temp = c(); targets_temp = c() ## Initializing vectors to store values
  
  Y = xV - Mt ## De-trending
  mean = mean(Y) ## Saving the mean value
  Z = Y - mean ## Zero-mean
  
  Z_f = fft(Z, inverse = FALSE) ## Converting Z to frequency domain
  
  g_index = which(is.na(x0)) ## Defining useful parameters
  for (i in 1:train_size){
    Z_p = Z_f * complex(modulus = runif(N, 1-Mod, 1+Mod), argument = runif(N, 0-Arg, 0+Arg)) ## Creating small perturbation
    Z_t = as.numeric(fft(Z_p, inverse = TRUE)) / N ## Converting back to time domain
    x_p = Z_t + mean + Mt ## Adding perturbed noise back to trend and periodic
    x_g = x_p; x_g[g_index] = 0 ## Imposing non-randomized gap structure
    inputs_temp = c(inputs_temp, x_g) ## Appending inputs
    targets_temp = c(targets_temp, x_p) ## Appending targets
  }
  
  inputs = array(matrix(inputs_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting targets
  #preprocessed = preprocess(inputs, targets) ## Preprocessing
  #inputs = preprocessed[[1]]; targets = preprocessed[[2]]
  return(list(inputs, targets, Mt, mean))
}


#' create_gaps
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. This function 
#' simulates the MCAR process to impose gaps into a time series according to a set combination of missing proportion and 
#' gap width. What differentiates this from Sophie's implementation us that the returned gapped series does not allow for gaps
#' to be placed where the original incomplete time series (x0) has missing data. This is for specific application in the 
#' autoencoder method for time series data imputation.
#' @param x {list}; List containing the complete time series to be imposed with missing values (equivalent to "x version")
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param p {int}; Proportion of data to be removed
#' @param g {float}; Width of missing data sections to be imposed on the series x
#' 
create_gaps <- function(x, x0, p, g){
  
  n <- length(x) ## Defining the number of data points
  to_remove = c() ## Initializing vector to store removal indices
  
  ## Some checks:
  stopifnot(is.numeric(x), is.numeric(x0), is.numeric(p), is.numeric(g), !is.null(x), !is.null(x0), !is.null(p), !is.null(g), 
            sum(is.na(x)) == 0, g %% 1 == 0, length(x) > 2,  p >= 0 & p <= (n-2)/n, g >= 0, p*g < length(x)-2)
  
  ## Creating list of possible indices to remove
  poss_values = which(!is.na(x0))
  if (1 %in% poss_values){poss_values = poss_values[-1]}
  if (n %in% poss_values){poss_values = poss_values[-length(poss_values)]}
  
  ## Determining number of data points to remove
  if ((p * n / g) %% 1 != 0) {
    warning(paste("Rounded to the nearest integer multiple; removed ", round(p*n/g,0)*g, " observations", sep =""))
  }
  
  if((p * n / g) %% 1 <= 0.5 & (p * n / g) %% 1 != 0) {
    end_while <- floor(p * n) - g
  } else {
    end_while <- floor(p * n)
  }
  
  ## Deciding which indices to remove
  num_missing <- 0
  while(num_missing < end_while) {
    
    start = sample(poss_values, 1)
    end = start + g - 1
    
    if (all(start:end %in% poss_values)){
      poss_values = poss_values[!poss_values %in% start:end]
      to_remove = c(to_remove, start:end)
      num_missing = num_missing + g
    }
  }
  
  ## Placing NA in the indices to remove
  x.gaps <- x
  x.gaps[to_remove] <- NA
  
  ## Sanity check
  x.gaps[1] <- x[1]
  x.gaps[n] <- x[n]
  
  ## Returning the final gappy data
  return(as.numeric(x.gaps))
}


#' preprocess
#' 
#' Function to preprocess the data before building the neural network imputer. The function fills NA values with
#' 0 which will be useful for the masking layer of the neural network architecture.
#' @param inputs {matrix}; Matrix object containing the input training data (i.e., an incomplete time series in each row)
#' @param targets {matrix}; Matrix object containing the target training data (i.e., a complete time series in each row)
#' 
# preprocess <- function(inputs, targets){
#   for (i in 1:dim(inputs)[1]){
#     inputs[i,] = ifelse(is.na(inputs[i,]), 0, inputs[i,])
#   }
#   return(list(inputs, targets))
# }


#' impute
#' 
#' Function to construct, compile, and fit a neural network model on the simulated training data
#' and then predict on the original time series. Completes the fifth step of the designed algorithm.
#' @param x0 {list}; List containing the original incomplete time series ("x naught")
#' @param inputs {matrix}; Matrix object containing the input training data
#' @param targets {matrix}; Matrix object containing the target training data
#' @param model {model}; Model architecture to be used for the neural network
#' 
imputer <- function(x0, inputs, targets, model){
  
  ## Defining useful parameters
  N = dim(inputs)[2]; EPOCHS = 50; BATCH_SIZE = 16
  
  x0 = matrix(ifelse(is.na(x0), 0, x0), ncol = 1); dim(x0) = c(1, N, 1) ## Formatting original series
  
  inputs = tf$constant(inputs) ## Creating input tensors
  targets = tf$constant(targets) ## Creating target tensors
  x0 = tf$constant(x0) ## Creating prediction tensor
  
  ## Compiling the model
  model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')
  
  ## Fitting the model to the training data
  model %>% fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                shuffle = FALSE, validation_split = 0.2)
  
  ## Predicting on the original series
  train_preds = model %>% predict(inputs)
  test_preds = model %>% predict(x0)
  return(list(train_preds, test_preds))
}


#' get_model
#' 
#' Function to return the desired TensorFlow neural network architecture.
#' @param Architecture {integer}; Desired architecture (numerically encoded)
#' @param N {integer}; Length of the original time series
#' 
get_model <- function(N){
  
  autoencoder = keras_model_sequential(name = 'Autoencoder') %>%
    layer_lstm(units = 256, input_shape = c(N, 1), return_sequences = TRUE, name = 'LSTM') %>%
    layer_dense(units = 256, activation = 'relu', name = 'encoder1') %>%
    layer_dense(units = 128, activation = 'relu', name = 'encoder2') %>%
    layer_dense(units = 64, activation = 'relu', name = 'encoder3') %>%
    layer_dense(units = 128, activation = 'relu', name = 'decoder1') %>%
    layer_dense(units = 256, activation = 'relu', name = 'decoder2') %>%
    layer_dense(units = 1, activation = 'linear', name = 'decoder3')
  return(autoencoder)
}


#' simulation_main
#' 
#' Function which facilitates the testing of the Neural Network Imputer (NNI) versus other methods implemented in the
#' interpTools package.
#' @param X {list}; List object containing a complete time series (should be scaled to (0,1))
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
simulation_main <- function(X, P, G, K, METHODS){
  
  ## Setting common seed
  set.seed(42)
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K); print('Imposed Gaps')
  
  ## Impute
  xI = simulation_impute(x0, METHODS); print('Interpolated Gaps')
  
  ## Evaluate
  performance = simulation_performance(X = X, xI = xI, x0 = x0)
  
  ## Aggregate
  aggregation = interpTools::aggregate_pf(performance)
  
  ## Return
  return(list(x0, xI, performance, aggregation))
}


#' simulation_run
#' 
#' Function to create a more organized method for running consecutive simulations. Calls the 'main' function to go through 
#' all neural network imputer steps and then returns simulation results. Takes all the same parameters as 'main.'
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#' @param method {string}; Method for data simulation in c('noise', 'all', 'separate')
#' @param Mod {float}; End point parameters for the 'runif' function when performing modulus perturbations (i.e., 1 +/- Mod)
#' @param Arg {float}; End point parameters for the 'runif' function when performing argument perturbations (i.e., 0 +/- Arg)
#'
simulation_run <- function(x0, max_iter, train_size, method, Mod, Arg, Architecture){
  
  ## Performing imputation with Neural Network Imputer (NNI)
  x_imp = main(x0 = x0,
               max_iter = max_iter, 
               train_size = train_size, 
               method = method, 
               Mod = Mod,
               Arg = Arg, 
               Architecture = Architecture)
  
  ## Returning imputed series
  return(x_imp)
}


#' simulation_impute
#' 
#' Function which acts as a wrapper to the interpTools interpolation process and also adds the ability to use the Neural 
#' Network Imputer (NNI). 
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
simulation_impute <- function(x0, METHODS){
  
  ## If the list contains NNI...
  if ('NNI' %in% METHODS){
    
    ## Removing NNI from methods list
    METHODS = METHODS[METHODS != 'NNI']
    
    ## Calling interpTools with remaining methods
    xI_all = interpTools::parInterpolate(x0, methods = METHODS)
    
    ## Performing NNI imputation
    xI_NNI = simulation_impute_NNI(x0)
    
    ## Joining the two results
    xI = list(c(xI_all[[1]], xI_NNI[[1]]))
  }
  
  else {
    
    ## Otherwise, just perform imputation with interpTools
    xI = interpTools::parInterpolate(x0, methods = METHODS)
  }
  
  ## Returning the imputed series
  return(xI)
}


#' simulation_impute_NNI
#' 
#' Function which acts as a wrapper to the NNI interpolation process. Takes an incomplete time series as input and uses NNI
#' to produce the interpolated series.
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#'
simulation_impute_NNI <- function(x0){
  
  ## Defining helpful variables
  D = 1
  M = 1
  P = length(x0[[1]])
  G = length(x0[[1]][[1]])
  K = length(x0[[1]][[1]][[1]])
  numCores = detectCores()
  
  ## Initializing lists to store interpolated series
  int_series = lapply(int_series <- vector(mode = 'list', M), function(x)
    lapply(int_series <- vector(mode = 'list', P), function(x) 
      lapply(int_series <- vector(mode = 'list', G), function(x) 
        x <- vector(mode = 'list', K))))
  
  int_data = list()
  
  ## Setting up the function call
  function_call = paste0("simulation_run(x, 5, 500, 'noise', 0.05, pi/6, 1)")
  
  ## Performing imputation
  int_series[[M]] = lapply(x0[[D]], function(x){
    lapply(x, function(x){
      lapply(x, function(x){
        eval(parse(text = function_call))})}
    )})
  
  ## Applying the function name 
  names(int_series) = c('NNI')
  
  ## Saving the imputed series
  int_data[[D]] = int_series
  
  ## Returning the imputed series
  return(int_data)
}


#' simulation_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to compute imputation performance with a variety of loss functions / performance metrics.
#' @param X {list}; List object containing the original complete time series
#' @param xI {list}; List object containing the interpolated time series
#' @param x0 {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
simulation_performance <- function(X, xI, x0, custom = NULL){
  
  D <- length(xI)
  M <- length(xI[[1]])
  P <- length(xI[[1]][[1]])
  G <- length(xI[[1]][[1]][[1]])
  K <- length(xI[[1]][[1]][[1]][[1]])
  
  # Initializing nested list object
  pf <- lapply(pf <- vector(mode = 'list', D),function(x)
    lapply(pf <- vector(mode = 'list', M),function(x) 
      lapply(pf <- vector(mode = 'list', P),function(x) 
        lapply(pf <- vector(mode = 'list', G),function(x)
          x<-vector(mode='list', K)))))
  
  prop_vec_names <- numeric(P)
  gap_vec_names <- numeric(G)
  method_names <- numeric(M)
  data_names <- numeric(D)
  
  prop_vec <- as.numeric(gsub("p","",names(xI[[1]][[1]])))
  gap_vec <- as.numeric(gsub("g","",names(xI[[1]][[1]][[1]])))
  method_names <- names(xI[[1]])
  
  if(is.null(names(xI))){
    data_names <- paste0("D", 1:D)
  }
  else{
    data_names <- names(xI)
  }
  
  # Evaluate the performance criteria for each sample in each (d,m,p,g) specification
  for(d in 1:D){
    for(m in 1:M){
      for(p in 1:P){
        prop_vec_names[p] <- c(paste("p", prop_vec[p],sep="")) # vector of names
        for(g in 1:G){
          gap_vec_names[g] <- c(paste("g", gap_vec[g],sep="")) # vector of names
          for(k in 1:K) { 
            #pf[[d]][[m]][[p]][[g]][[k]] <- unlist(eval_performance(x = OriginalData[[d]], X = IntData[[d]][[m]][[p]][[g]][[k]], gappyx = GappyData[[d]][[p]][[g]][[k]], custom = custom))
            
            ## Editing here -----------
            ## Removing the "[[d]]" from OriginalData
            ## Changing to "my_eval_performance" after changing the function names from their originals
            pf[[d]][[m]][[p]][[g]][[k]] <- unlist(simulation_eval_performance(x = X, X = xI[[d]][[m]][[p]][[g]][[k]], gappyx = x0[[d]][[p]][[g]][[k]], custom = custom))
          }
          names(pf[[d]][[m]][[p]]) <- gap_vec_names
        }
        names(pf[[d]][[m]]) <- prop_vec_names
      }
      names(pf[[d]]) <- method_names
    }
    names(pf) <- data_names
  }
  
  pf <- lapply(pf, function(x) 
    lapply(x, function(x) 
      lapply(x, function(x)
        lapply(x, function(x){
          logic <- unlist(lapply(x,FUN = function(x) !is.null(x)))
          x <-x[logic]
        }))))
  
  class(pf) <- "pf"
  return(pf) 
}


#' simulation_eval_performance
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to aggregate imputer performance over several iterations and combinations of P, G, and K and return
#' a neatly laid out performance report.
#' @param x {list}; List object containing the original complete time series
#' @param X {list}; List object containing the interpolated time series
#' @param GappyData {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
simulation_eval_performance <- function(x, X, gappyx, custom = NULL) {
  
  # x = original , X = interpolated 
  
  if(sum(is.na(gappyx)) == 0) stop(paste0("Gappy data in 'gappyx' does not contain NAs. Please impose gaps and try again."))
  if(sum(x - gappyx, na.rm = TRUE) != 0) stop(paste0("Gappy data in 'gappyx' is not representative of 'x' (original data). The two vectors are non-conforming."))
  #if(sum(X[which(!is.na(gappyx))] - x[which(!is.na(gappyx))]) != 0) stop(paste0("Non-interpolated points in 'X' do not match those of the original data in 'x'.  The two vectors are non-conforming."))
  
  if(!is.null(X)){
    stopifnot((is.numeric(x) | is.null(x)),
              (is.numeric(X) | is.null(X)),
              (is.numeric(gappyx) | is.null(gappyx)),
              length(x) == length(X),
              length(gappyx) == length(x), 
              length(gappyx) == length(X))
    
    # identify which values were interpolated
    index <- which(is.na(gappyx))
    
    # only consider values which have been replaced
    X <- X[index]
    x <- x[index]
    
    n <- length(x)
    
    return <- list()
    
    # Coefficent of Correlation, r
    numerator <- sum((X - mean(X))*(x - mean(x)))
    denominator <- sqrt(sum((X - mean(X))^2)) * sqrt(sum((x - mean(x))^2))
    return$pearson_r <- numerator / denominator
    
    # r^2
    return$r_squared <- return$pearson_r^2  
    
    # Absolute Differences
    return$AD <- sum(abs(X - x))
    
    # Mean Bias Error 
    return$MBE <- sum(X - x) / n
    
    # Mean Error 
    return$ME <- sum(x - X) / n
    
    # Mean Absolute Error 
    return$MAE <- abs(sum(x - X)) / length(x)
    
    # Mean Relative Error 
    if (length(which(x == 0)) == 0) {
      return$MRE <- sum((x - X) / x)  
    } else {
      return$MRE <- NA
    }
    
    # Mean Absolute Relative Error ##### Lepot
    if (length(which(x == 0)) == 0) {
      return$MARE <- 1/length(x)*sum(abs((x - X) / x))
    } else {
      return$MARE <- NA 
    }
    
    # Mean Absolute Percentage Error 
    return$MAPE <- 100 * return$MARE
    
    # Sum of Squared Errors
    return$SSE <- sum((X - x)^2)
    
    # Mean Square Error 
    return$MSE <- 1 / n * return$SSE
    
    # Root Mean Squares, or Root Mean Square Errors of Prediction 
    if (length(which(x == 0)) == 0) {
      return$RMS <- sqrt(1 / n * sum(((X - x)/x)^2))
    } else {
      return$RMS <- NA 
    }
    
    # Mean Squares Error (different from MSE, referred to as NMSE)
    return$NMSE <- sum((x - X)^2) / sum((x - mean(x))^2)
    
    # Reduction of Error, also known as Nash-Sutcliffe coefficient 
    return$RE <- 1 - return$NMSE
    
    # Root Mean Square Error, also known as Root Mean Square Deviations
    return$RMSE <- sqrt(return$MSE)
    
    # Normalized Root Mean Square Deviations 
    return$NRMSD <- 100 * (return$RMSE / (max(x) - min(x)))
    
    # Root Mean Square Standardized Error 
    if (sd(x) != 0) {
      return$RMSS <- sqrt(1 / n * sum(( (X-x)/sd(x) )^2))  
    } else {
      return$RMSS <- NA 
    }
    
    # Median Absolute Percentage Error
    if (length(which(x == 0)) == 0) {
      return$MdAPE <- median(abs((x - X) / x))*100  
    } else {
      return$MdAPE <- NA
    }
    
    ######### ADDITIONS:
    ## Log-Cosh Loss
    return$LCL <- sum(log(cosh(X - x))) / n
    
    ## Quantile Loss
    ## ---
    
    
    # Custom functions
    
    if(!is.null(custom)){
      
      ####################
      ### LOGICAL CHECKS
      ####################
      
      n_custom <- length(custom)
      
      if(n_custom == 1){
        is_fn <- !inherits(try(match.fun(custom), silent = TRUE), "try-error") # FALSE if not a function
      }
      else if(n_custom > 1){
        is_fn <- logical(n_custom)
        for(k in 1:n_custom){
          is_fn[k] <- !inherits(try(match.fun(custom[k]), silent = TRUE), "try-error") # FALSE if not a function
        }
      }
      
      if(!all(is_fn)){
        not <- which(!is_fn)
        stop(c("Custom function(s): ", paste0(custom[not], sep = " ") ,", are not of class 'function'."))
      }
      
      # Check that the output of the function is a single value
      
      check_single <- function(fn){
        
        x <- rnorm(10)
        X <- rnorm(10)
        
        val <- match.fun(fn)(x = x, X = X)
        
        return(all(length(val) == 1, is.numeric(val)))
      }
      
      logic <- logical(n_custom)
      
      for(k in 1:n_custom){
        logic[k] <- check_single(custom[k])
      }
      
      if(!all(logic)){
        stop(c("Custom function(s): ", paste0(custom[!logic], sep = " "), ", do not return a single numeric value."))
      }
      
      # Computing custom metric values
      
      return_call <- character(n_custom)
      
      for(k in 1:n_custom){
        
        return_call[k] <- paste0("return$",custom[k]," <- match.fun(",custom[k],")(x = x, X = X)")
        
        eval(parse(text = return_call[k]))
      }
    }
    return(return)
  }
  else if(is.null(X)){
    return(NULL)
  }
}


#' simulation_heatmap
#' 
#' Function which takes an aggregations object as input and produces a multi-heatmap plot comparing 
#' performance across all methods. The user can specify the specific metric and aggregation function
#' to use (i.e., mean, median, IQR, etc.).
#' @param agg {aggregate_pf}; Aggregation object containing simulation results
#' @param P {list}; List of missing proportions to consider for the visualization
#' @param G {list}; List of gap width structures to consider for the visualization
#' @param crit {string}; Metric to consider from the aggregation object
#' @param f {string}; Aggregation type to be considered
#' @param title {string}; Any additional text to be added on to the end of the default title
#' @param colors {list}; List of plot colors to be considered for the visualization
#'
simulation_heatmap <- function(agg, P, G, METHODS, crit = 'MAE', f = 'median', title = '', 
                               colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")){
  
  ## Initializing a data-frame 
  data = data.frame()
  
  ## Defining full color palette
  col = colorRampPalette(colors = colors)(100)
  
  ## Adding data from all aggregations
  for (p in P){
    for (g in G){
      for (method in METHODS){
        temp = eval(parse(text = paste0('as.data.frame(agg$D1$p', p, '$g', g, '$', method, ')')))
        temp$metric = rownames(temp); rownames(temp) = NULL
        data = rbind(data, temp)
      }
    }
  }
  ## Filtering the data-frame
  data = eval(parse(text = paste0('data %>% dplyr::select(method, prop_missing, gap_width, metric, ', f, ')'))) %>% 
    dplyr::filter(metric == crit) %>%
    dplyr::rename('P' = 'prop_missing', 'G' = 'gap_width')
  
  ## Creating plot
  theme_update(plot.title = element_text(hjust = 0.5))
  ggplot(data, aes(as.factor(P), as.factor(G), fill = data[,5])) +
    geom_tile(color = "gray95", lwd = 0.5, linetype = 1) +
    facet_grid(~ method) +
    labs(title = paste0('Simulation Results ', title), 
         x = "Missing Proportion (P)", 
         y = "Gap Width (G)", 
         fill = data[1,4]) +
    scale_fill_gradientn(colours = col, values = c(0,1)) + 
    guides(fill = guide_colourbar(label = TRUE, ticks = TRUE)) +
    geom_text(aes(label = round(data[,5], 3)), color = "black", size = 3) +
    theme_minimal() + 
    theme(plot.title = element_text(hjust = 0.5, size = 18)) +
    theme(strip.text = element_text(face = 'bold', size = 10))
}


#' sim_Tt_mod
#' 
#' Function to simulate a time series with a modulated signal. Needs to be updated.
#' @param N {aggregate_pf}; Length of the output series
#' @param numFreq {integer}; Number of frequency components 
#' @param P {integer}; Desired number of cosine components
#'
sim_Tt_mod <- function(N = 1000, numFreq, P){
  
  ## Defining helpful variables
  Tt_list <- list()
  t = 0:(N-1)
  fourierFreq = 2*pi/N
  P = P + 1
  text = ''
  
  for (i in 1:numFreq){
    
    mu = round(rnorm(1, mean = 0, sd = N/200), 3)
    f = round(runif(1, fourierFreq, pi), 3)
    a = round(rnorm(P, mean = 0, sd = N/200), 3)
    
    text = paste0(text, "(", mu, ")*cos(2*", round(pi, 3), "*(", f, "*t + ", sep = "", collapse = "")
    
    for (p in 1:P){
      text = paste0(text, a[p] / (p), "*t^", p, " + ", sep = "", collapse = "")
    }
    
    text = paste0(text, '0)) + ')
  }
  
  text = substr(text, start = 1, stop = nchar(text) - 3)
  
  print(text)
  
  ## Returning final list object
  Tt_list$fn <- paste(text, collapse="")  
  Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
  return(Tt_list)
}






## Example experiments
## -----------------------


# ## Reading the data files:
# births = read.csv('Data/daily-total-female-births.csv')
# temp = read.csv('Data/daily-max-temperatures.csv')
# sunspots = read.csv('Data/monthly-sunspots.csv')
# 
# ## Selecting columns of interest
# births = births$Births
# temp = temp$Temperature
# sunspots = sunspots$Sunspots
# 
# ## Creating sample plots of the data
# plot(births, type = 'l', lwd = 2, xlab = 'Date', ylab = 'Female Births'); grid()
# plot(temp, type = 'l', lwd = 2, xlab = 'Date', ylab = 'Temperature'); grid()
# plot(sunspots, type = 'l', lwd = 2, xlab = 'Date', ylab = 'Sunspots'); grid()
# 
# ## Defining experiment parameters
# P = c(0.1, 0.2, 0.3)
# G = c(10, 25, 50)
# METHODS = c('HWI', 'LI', 'NNI')
# 
# 
# ## Births:
# births_results = FRAMEWORK(births, P, G, K = 25, METHODS)
# x0 = births_results[[1]]
# xI = births_results[[2]]
# performance = births_results[[3]]
# aggregation = births_results[[4]]
# 
# births_heatmap_lcl = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'LCL', title = '(Births Data, N = 365)')
# births_heatmap_mae = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'MAE', title = '(Births Data, N = 365)')
# births_heatmap_rmse = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'RMSE', title = '(Births Data, N = 365)')
# 
# births_heatmap_lcl
# births_heatmap_mae
# births_heatmap_rmse
# 
# 
# ## Temperature:
# temp_results = FRAMEWORK(temp, P, G, K = 25, METHODS)
# x0 = temp_results[[1]]
# xI = temp_results[[2]]
# performance = temp_results[[3]]
# aggregation = temp_results[[4]]
# 
# temp_heatmap_lcl = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'LCL')
# temp_heatmap_mae = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'MAE')
# temp_heatmap_rmse = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'RMSE')
# 
# temp_heatmap_lcl
# temp_heatmap_mae
# temp_heatmap_rmse
# 
# 
# ## Sunspots:
# sunspots_results = FRAMEWORK(sunspots, P, G, K = 25, METHODS)
# x0 = sunspots_results[[1]]
# xI = sunspots_results[[2]]
# performance = sunspots_results[[3]]
# aggregation = sunspots_results[[4]]
# 
# sunspots_heatmap_lcl = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'LCL', title = '(Sunspots Data, N = 2820)')
# sunspots_heatmap_mae = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'MAE', title = '(Sunspots Data, N = 2820)')
# sunspots_heatmap_rmse = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'RMSE', title = '(Sunspots Data, N = 2820)')
# 
# sunspots_heatmap_lcl
# sunspots_heatmap_mae
# sunspots_heatmap_rmse
# 
# 
# ## Simple:
# set.seed(1)
# X = interpTools::simXt(N = 500, numTrend = 0, mu = 0, numFreq = 2)$Xt
# 
# results = FRAMEWORK(X, P, G, K = 25, METHODS)
# x0 = results[[1]]
# xI = results[[2]]
# performance = results[[3]]
# aggregation = results[[4]]
# 
# results_heatmap_lcl = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'LCL', title = '(Sim. Data, N = 500)')
# results_heatmap_mae = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'MAE', title = '(Sim. Data, N = 500)')
# results_heatmap_rmse = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'RMSE', title = '(Sim. Data, N = 500)')
# 
# results_heatmap_lcl
# results_heatmap_mae
# results_heatmap_rmse
# 
# 
# ## Simple (but modulated):
# set.seed(1)
# X = simTt_mod(N = 500, numFreq = 8, P = 3)$value
# 
# modulated_results = FRAMEWORK(X, P, G, K = 25, METHODS)
# x0 = modulated_results[[1]]
# xI = modulated_results[[2]]
# performance = modulated_results[[3]]
# aggregation = modulated_results[[4]]
# 
# results_heatmap_lcl = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'LCL', title = '(Sim. Modulated Data, N = 500)')
# results_heatmap_mae = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'MAE', title = '(Sim. Modulated Data, N = 500)')
# results_heatmap_rmse = my_new_multiHeatmap(aggregation, P, G, METHODS, f = 'mean', crit = 'RMSE', title = '(Sim. Modulated Data, N = 500)')
# 
# results_heatmap_lcl
# results_heatmap_mae
# results_heatmap_rmse
# 
# 
# ## Plotting some experiment results
# x_0 = x0[[1]]$p0.3$g50[[10]]
# x_i = xI[[1]]$HWI$p0.3$g50[[10]]
# x_li = xI[[1]]$LI$p0.3$g50[[10]]
# Xt_est = estimator(x_li, method = 'Xt')
# 
# plot(x_i, type = 'l', col = 'red')
# lines(x_li, type = 'l', col = 'green')
# lines(Xt_est, type = 'l', col = 'dodgerblue')
# lines(sunspots, type = 'l', lwd = 2); grid()







