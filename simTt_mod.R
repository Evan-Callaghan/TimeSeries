



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










simTt_mod <- function(N = 1000, mu = NULL, f = NULL){
  
  ## Defining helpful variables
  Tt_list <- list()
  t = 0:(N-1)
  text = ''
  
  if (is.null(mu)){
    mu = 2
  }
  
  if (is.null(f)){
    f = runif(1, 1/N, 0.5)
  }
  
  #text = paste0(mu, "*cos(2*pi*(", f, '*t))')
  text = paste0(mu, "*cos(2*pi*(", f, '*t + sin(t)))')
  
  ## Returning final list object
  Tt_list$fn <- paste(text, collapse="")  
  Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
  return(Tt_list)
}

Tt = simTt_mod(N = 100, f = 0.1)
print(Tt$fn)
plot(0:99, Tt$value, type = 'l'); grid()




