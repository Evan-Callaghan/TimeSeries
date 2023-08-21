## Blocking function
library(dplyr)
library(interpTools)


blocking <- function(x, window, forecast){
  
  ## Checking for validity of the series
  if (sum(is.na(x)) == length(x)){print('Entire series is NA.'); return(NULL)}
  if (is.na(x[1])){print('Remove leading NA values from the series.'); return(NULL)}
  if (is.na(x[length(x)])){print('Remove ending NA values from the series.'); return(NULL)}
  if (length(x) < (window+forecast)){print('Series does not meet length requirement.'); return(NULL)}
  if (sum(is.na(x[1:(window+forecast)])) > 0){print('Series requires more data points before first NA value.'); return(NULL)}
  if (sum(is.na(x[(length(x)-window):length(x)])) > 0){print('Series requires more data points after last NA value.'); return(NULL)}
  
  ## Defining parameters and initializing vector to store results
  N = length(x)
  block = c()
  condition = TRUE; i = 1; M = 0
  
  while(condition){
    
    ## When the current value is not null:
    if (!is.na(x[i])){
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (!is.na(x[i])){next}
        else{end_idx = i - 1; continue = FALSE}
        }
      M = M + 1; seq_info = c(start_idx, end_idx, end_idx - start_idx + 1, 1)
      block = c(block, seq_info)
      }
    
    ## When the current value is null:
    else{
      continue = TRUE
      start_idx = i
      while(continue){
        i = i + 1
        if (is.na(x[i]) & i <= N){next}
        else{end_idx = i - 1; continue = FALSE}
      }
      M = M + 1; seq_info = c(start_idx, end_idx, end_idx - start_idx + 1, 0)
      block = c(block, seq_info)
    }
    if (i > N){condition = FALSE}
  }
  ## Formatting the final block
  block = as.data.frame(matrix(block, nrow = M, byrow = TRUE), dim = c(M, 4)) %>%
    dplyr::rename(Start = V1, End = V2, Gap = V3, Complete = V4)
  block$Complete = ifelse(block$Complete == 0, FALSE, TRUE)
  block$Valid = ifelse(block$Complete == FALSE, NA, 
                       ifelse(block$Gap >= (window + forecast), TRUE, FALSE))
  return(block)
}


x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, NA, NA, NA, 13, 14, 15, 16, 17, 18, 19, 20)
blocking(x, window = 2, forecast = 1)

x = c(NA, NA, 1, 2, 3, 4, 5, 6, 7, 8)
blocking(x, window = 2, forecast = 1)

x = c(10, 11, NA, NA, 2, 3, 4, NA, NA)
blocking(x, window = 2, forecast = 1)

x = c(1, 2)
blocking(x, window = 2, forecast = 1)

x = c(NA, NA, NA, NA, NA, NA, NA)
blocking(x, window = 2, forecast = 1)

x = c(1, 2, NA, NA, 3, 4, 5, 6, 7, 8)
blocking(x, window = 2, forecast = 1)

x = c(1, 2, 3, NA, NA, 3, 4, 5, 6, NA, 7, 8)
blocking(x, window = 2, forecast = 1)



set.seed(10)
X = interpTools::simXt(N = 100)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.3, g = 5, K = 1)[[1]]$p0.3$g5[[1]]
X0

my_block = blocking(X0, window = 3, forecast = 1)
my_block

training_index = which(my_block$Valid == TRUE)
prediction_index = which(is.na(my_block$Valid))





  