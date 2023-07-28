



simTt_mod <- function(N = 1000, numFreq, P){
  
  ## Defining helpful variables
  Tt_list <- list()
  t = 0:(N-1)
  fourierFreq = 2*pi/N
  P = P + 1
  text = ''
  
  for (i in 1:numFreq){
    
    #mu = rnorm(1, mean = 0, sd = N/200)
    mu = 2
    f = runif(1, fourierFreq, 0.5)
    a = rnorm(P, mean = 0, sd = N/200)
    
    text = paste0(text, "(", mu, ")*cos(2*pi*(", f, "*t + ", sep = "", collapse = "")
    
    for (p in 1:P){
      text = paste0(text, "(", a[p] / (p), ")*t^", p, " + ", sep = "", collapse = "")
    }
    
    text = paste0(text, '0)) + ')
  }
  
  text = substr(text, start = 1, stop = nchar(text) - 3)
  
  ## Returning final list object
  Tt_list$fn <- paste(text, collapse="")  
  Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
  return(Tt_list)
}








# 
# 
# simTt_mod <- function(N = 1000, mu = NULL, f = NULL){
#   
#   ## Defining helpful variables
#   Tt_list <- list()
#   t = 0:(N-1)
#   text = ''
#   
#   if (is.null(mu)){
#     mu = 2
#   }
#   
#   if (is.null(f)){
#     f = runif(1, 1/N, 0.5)
#   }
#   
#   #text = paste0(mu, "*cos(2*pi*(", f, '*t))')
#   text = paste0(mu, "*cos(2*pi*(", f, '*t + sin(t)))')
#   
#   ## Returning final list object
#   Tt_list$fn <- paste(text, collapse="")  
#   Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
#   return(Tt_list)
# }
# 
# Tt = simTt_mod(N = 100, f = 0.1)
# print(Tt$fn)
# plot(0:99, Tt$value, type = 'l'); grid()
# 
# 
# 
# 
# 
# 
# source('initialize.R')
# source('estimate.R')
# 
# 
# set.seed(12)
# X = interpTools::simXt(N = 500, numTrend = 2, numFreq = 2, b = c(10, 0.5), w = c(0.5, 10))$Xt
# X0 = interpTools::simulateGaps(list(X), p = 0.2, g = 50, K = 1)[[1]]$p0.2$g50[[1]]
# 
# ## Plot 1: Original and gapped time series 
# plot(X, type = 'l', lty = 3); grid()
# lines(X0, type = 'l', lwd = 2)
# 
# ## Plot 2: Initialize Step
# XI = initialize(X0)
# plot(XI, type = 'l', col = 'red', lty = 3)
# lines(X0, type = 'l', lwd = 2); grid()
# 
# ## Plot 3a: Estimate Trend Step
# XE_Mt = estimator(XI, method = 'Mt')
# plot(XI, type = 'l', col = 'red', lty = 3)
# lines(X0, type = 'l', lwd = 2); grid()
# lines(XE_Mt[,1], type = 'l', lty = 2, col = 'dodgerblue')
# 
# ## Plot 3b: Estimate Periodic Step
# XE_Tt = estimator(XI - XE_Mt, method = 'Tt')
# plot(XI - XE_Mt, type = 'l', lwd = 2)
# lines(XE_Tt, type = 'l', col = 'dodgerblue')
# 
# ## Plot 3c: Estimate Both Together
# XE = estimator(XI, method = 'Xt')
# plot(XI, type = 'l', col = 'red', lty = 3)
# lines(X0, type = 'l', lwd = 2); grid()
# lines(XE, type = 'l', lty = 2, col = 'dodgerblue')
# 
# 


