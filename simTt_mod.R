



simTt_mod <- function(N = 1000, numFreq, P){
  
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
  
  ## Returning final list object
  Tt_list$fn <- paste(text, collapse="")  
  Tt_list$value <- eval(parse(text = paste(text, collapse = "")))
  return(Tt_list)
}