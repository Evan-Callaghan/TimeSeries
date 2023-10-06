#############################################
## Autoencoder Simulation Helper Functions ##
#############################################


## Defining all functions
## -----------------------


#' main
#' 
#' Function that works through the designed algorithm and calls functions in order.
#' @param x0 {list}; List object containing the original incomplete time series
#' @param max_iter {integer}; Maximum number of iterations of the algorithm to perform
#' @param train_size {integer}; Number of new time series to construct
#'
main <- function(x0, max_iter, train_size){
  
  ## Defining useful variables
  N = length(x0)
  
  ## Defining matrix to store results
  results = matrix(NA, ncol = N, nrow = max_iter)
  
  ## Building the neural network model
  model = get_model(N)
  
  ## Step 1: Estimating p and g
  p = estimate_p(x0); g = estimate_g(x0)
  
  ## Step 2: Linear imputation
  xV = initialize(x0)
  
  for (i in 1:max_iter){
    
    ## Steps 3/4: Simulating time series and imposing gap structure
    data = simulator(x0, xV, p, g, i, train_size)
    inputs = data[[1]]; targets = data[[2]]
    
    ## Step 5: Performing the imputation
    preds = imputer(x0, inputs, targets, model)
    
    ## Step 6: Extracting the predicted values and updating imputed series
    xV = ifelse(is.na(x0), preds[1,,1], x0); results[i,] = xV
  }
  
  ## Returning a point estimation for each missing data point
  if (max_iter == 1){
    return(results[1,])
  }
  
  ## Returning a distribution for each missing data point
  else{
    return(results) 
  }
}


#' estimate_p
#' 
#' Function that returns the proportion of missing data in the input series. Simply 
#' returns the number of NA values divided by the length.
#' @param x0 {list}; List object containing the original incomplete time series
#'
estimate_p <- function(x0){
  N = length(x0)
  return(round(sum(is.na(x0))/N, 2))
}


#' estimate_g
#'
#' Function that returns the estimated width of missing data gaps in the input series.
#' Finds the length of all blocks of missing data and returns the mode of that list.
#' @param x0 {list}; List object containing the original incomplete time series
#'
estimate_g <- function(x0){
  
  ## Defining helpful parameters
  condition = TRUE
  N = length(x0); i = 1
  
  ## Initializing vector to store results
  widths = c()
  
  while(condition){
    
    if (!is.na(x0[i])){
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (!is.na(x0[i])){next}
        else{end_idx = i - 1; continue = FALSE}
      }
    }
    
    else{
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (is.na(x0[i]) & i <= N){next}
        else{end_idx = i - 1; continue = FALSE}
      }
      widths = c(widths, end_idx - start_idx + 1)
    }
    if (i > N){condition = FALSE}
  }
  
  ## Computing g
  u_widths = unique(widths)
  g = u_widths[which.max(tabulate(match(widths, u_widths)))]
  
  return(g)
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


#' simulator
#' 
#' Function to generate a set of training data for the network by randomly imposing 
#' the specified gap structure into the original incomplete time series.
#' 
#' @param x0 {list}; List containing the original incomplete time series ("x naught") 
#' @param xV {list}; List containing the current version of imputed series ("x version")
#' @param p {float}; Proportion of data to be removed
#' @param g {integer}; Width of missing data sections
#' @param iteration {integer}; Current iteration of the autoencoder algorithm
#' @param train_size {integer}; Number of new time series to construct
#' 
simulator <- function(x0, xV, p, g, iteration, train_size){
  
  N = length(xV); n_missing = N * p ## Defining useful parameters
  inputs_temp = c(); targets_temp = c(); weights_temp = c() ## Initializing vectors to store values
  
  for (i in 1:train_size){
    
    set.seed((iteration * train_size) + i) ## Setting a common seed
    x_g = create_gaps(xV, x0, p, g) ## Imposing randomized gap structure
    
    inputs_temp = c(inputs_temp, x_g) ## Appending inputs
    targets_temp = c(targets_temp, xV) ## Appending targets
  }
  
  inputs = array(matrix(inputs_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting inputs
  targets = array(matrix(targets_temp, nrow = train_size, byrow = TRUE), dim = c(train_size, N, 1)) ## Properly formatting targets
  return(list(inputs, targets))
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
#' @param p {float}; Proportion of data to be removed
#' @param g {integer}; Width of missing data sections
#' 
create_gaps <- function(x, x0, p, g){
  
  n = length(x) ## Defining the number of data points
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
  num_missing = 0
  while(num_missing < end_while) {
    
    start = sample(poss_values, 1)
    end = start + g - 1
    
    if (all(start:end %in% poss_values)){
      poss_values = poss_values[!poss_values %in% start:end]
      to_remove = c(to_remove, start:end)
      num_missing = num_missing + g
    }
  }
  
  ## Placing NA in the indices to remove (represented by 0)
  x.gaps = x
  x.gaps[to_remove] = 0
  
  ## Sanity check
  x.gaps[1] = x[1]
  x.gaps[n] = x[n]
  
  ## Returning the final gappy data
  return(as.numeric(x.gaps))
}


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
  N = dim(inputs)[2]; EPOCHS = 30; BATCH_SIZE = 32
  
  x0 = matrix(ifelse(is.na(x0), 0, x0), ncol = 1); dim(x0) = c(1, N, 1) ## Formatting original series
  
  inputs = tf$constant(inputs) ## Creating input tensors
  targets = tf$constant(targets) ## Creating target tensors
  x0 = tf$constant(x0) ## Creating prediction tensor
  
  ## Compiling the model
  model %>% compile(optimizer = 'adam', loss = 'MeanSquaredError')
  
  ## Fitting the model to the training data
  model %>% fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                shuffle = FALSE, validation_split = 0.2, verbose = 0)
  
  ## Predicting on the original series
  preds = model %>% predict(x0, verbose = 0)
  return(preds)
}


#' get_model
#' 
#' Function to return the desired TensorFlow neural network architecture.
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
    layer_dense(units = 1, name = 'decoder3')
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
  x0 = interpTools::simulateGaps(list(X), P, G, K)
  
  ## Impute
  xI = simulation_impute(x0)
  
  ## Evaluate
  performance = simulation_performance(X = X, xI = xI, x0 = x0)
  
  ## Save
  results = simulation_saver(performance, P, G, K, METHODS)
  
  ## Return
  return(results)
}


#' simulation_impute
#' 
#' Function which acts as a wrapper to the interpTools interpolation process and also adds the ability to use the Neural 
#' Network Imputer (NNI). 
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#'
simulation_impute <- function(x0){
  
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
  function_call = paste0("main(x0 = x, max_iter = 1, train_size = 320)")
  
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
            pf[[d]][[m]][[p]][[g]][[k]] <- unlist(simulation_performance_helper(x = X, X = xI[[d]][[m]][[p]][[g]][[k]], gappyx = x0[[d]][[p]][[g]][[k]], custom = custom))
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


#' simulation_performance_helper
#' 
#' Function taken directly from Sophie's implementation of the 'interpTools' package and then slightly edited. 
#' Function to aggregate imputer performance over several iterations and combinations of P, G, and K and return
#' a neatly laid out performance report.
#' @param x {list}; List object containing the original complete time series
#' @param X {list}; List object containing the interpolated time series
#' @param GappyData {list}; List object containing the original incomplete time series
#' @param custom {function}; Customized loss function / performance criteria if desired
#'
simulation_performance_helper <- function(x, X, gappyx, custom = NULL) {
  
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


#' simulation_saver
#' 
#' Function to store all simulation results in a consistent format. Takes the simulation performance as input
#' and sorts through the nested lists to organize imputation performance in a data-frame object. The returned 
#' data-frame can be easily exported to save simulation results.
#' @param performance {list}; Nested list object containing performance evaluation from the the set of simulations
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
simulation_saver <- function(performance, P, G, K, METHODS){
  
  ## Initializing data-frame to store results
  performance_summary = data.frame()
  
  ## Looping through all simulation combinations
  for (method in METHODS){
    for (p in P){
      for (g in G){
        for (k in 1:K){
          
          ## Appending the selected performance metrics
          metrics = paste0("performance$D1$", method, "$p", p, "$g", g, "[[", k, "]]")
          performance_summary = rbind(performance_summary, c(method, p, g, k, as.numeric(eval(parse(text = metrics)))))
        }
      }
    }
  }
  ## Cleaning the final data-frame
  colnames(performance_summary) = c('Method', 'P', 'G', 'K', 'pearson_r', 'r_squared', 'AD', 'MBE', 'ME', 'MAE', 'MRE', 'MARE',
                                    'MAPE', 'SSE', 'MSE', 'RMS', 'NMSE', 'RE', 'RMSE', 'NRMSD', 'RMSS', 'MdAPE', 'LCL')
  return(performance_summary)
}


#' plot_ts
#' 
#' Function to formalize the time series plotting process. Follows a standard form for which all time 
#' series plots/visualization will be displayed in the thesis.
#' @param x {list}; List object containing the time series to be plotted
#' @param title {string}; Title of the plot (default is empty)
#'
plot_ts <- function(x, title = ''){
  
  X_t = data.frame(index = seq(1, length(x)), value = x)
  
  plt = ggplot(data = X_t, aes(x = index, y = value)) +
    geom_line(color = "#476d9e", linewidth = 0.8) + 
    geom_point(color = '#476d9e', size = 0.4) +
    labs(title = paste0(title), x = "Index", y = "Value") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 20), 
          axis.text = element_text(color = 'black', size = 10), 
          axis.title.x = element_text(color = 'black', size = 14, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 14, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'grey70', linewidth = 0.5, linetype = 'dotted'),
          text=element_text(family="Helvetica"))
  return(plt)
}


#' simulation_plot
#' 
#' Function to formalize the imputation simulation plotting process. Follows a standard form for which 
#' all time series imputation plots/visualization will be displayed in the thesis.
#' @param aggregation {aggregation_pf}; Aggregation of imputation performance across methods
#' @param criteria {string}; Desired criteria for imputation performance evaluation (default 'RMSE)
#' @param agg {string}; Function type for aggregating data across K iterations (default 'mean')
#' @param title {string}; Desired title for the returned plot
#' @param levels {numeric}; Vector containing the method names in the desired order for the plot
#'
simulation_plot <- function(aggregation, criteria = 'RMSE', agg = 'mean', title = '', levels){
  
  ## Initializing data-frame to store results
  data = data.frame()
  
  ## Creating structured data-frame
  for (p in P){
    for (g in G){
      for (method in METHODS){
        temp = eval(parse(text = paste0('as.data.frame(aggregation$D1$p', p, '$g', g, '$', method, ')')))
        temp$metric = rownames(temp); rownames(temp) = NULL
        data = rbind(data, temp)
      }
    }
  }
  
  ## Cleaning the data-frame
  data = data %>% dplyr::select(method, gap_width, prop_missing, metric, all_of(agg)) %>%
    dplyr::filter(metric == criteria) %>%
    dplyr::rename('P' = 'prop_missing', 'G' = 'gap_width', 'value' = all_of(agg)) %>%
    dplyr::arrange(desc(method))
  
  ## Creating colour palette
  colors = c("#91cff2", "#85bee4", "#78add5", "#6c9dc7", "#608cb9", "#537dab", "#476d9e", "#3b5e90", "#2e4f83", "#204176")
  col = colorRampPalette(colors = colors)(100)
  
  ## Creating plot
  plt = ggplot(data, aes(as.factor(P), as.factor(G), fill = value)) +
    geom_tile(color = '#204176', linewidth = 0.1) +
    facet_grid(~ factor(method, levels = levels)) +
    labs(title = paste0(title), 
         x = "Missing Proportion (P)", 
         y = "Gap Width (G)", 
         fill = data$metric[1]) +
    scale_fill_gradientn(colours = col, values = c(0,1)) + 
    guides(fill = guide_colourbar(label = TRUE, ticks = TRUE, title = data$metric[1])) +
    geom_text(aes(label = round(value, 3)), color = "white", size = 4) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0, face = 'bold', size = 18), 
          strip.background = element_rect(fill = 'white'), 
          strip.text = element_text(color = 'black', face = 'bold', size = 12), 
          panel.spacing = unit(0.8, 'lines'), 
          axis.text = element_text(color = 'black', size = 12), 
          axis.title.x = element_text(color = 'black', size = 14, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 14, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'white'), 
          legend.box.background = element_rect(),
          legend.box.margin = margin(4, 4, 4, 4), 
          legend.position = "right", 
          legend.key.height = unit(1, 'cm'), 
          legend.title.align = -0.5)
  return(plt)
}






