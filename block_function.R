## Blocking function



library(dplyr)
library(interpTools)
library(tsinterp)

block_function <- function(x, window, forecast){
  
  ## Getting missing block from findBlock' function
  mask = ifelse(is.na(x), NA, TRUE)
  missing_block = findMissingBlock(mask)
  
  ## Defining helpful parameters and initializing data-frame
  N = length(X0)
  M = dim(missing_block)[1] + 1
  training_block = data.frame(Start=rep(NA,M), End=rep(NA,M), Gap=rep(NA,M), Train=rep(NA,M))
  
  ## Setting values in training_block for each of the M intervals
  for (i in 1:M){
    if (i == 1){
      training_block[i,1:2] = c(1, missing_block$Start[1]-1) 
    }
    else if (i == M){
      training_block[i,1:2] = c(missing_block$End[i-1]+1, N)
    }
    else{
      training_block[i,1:2] = c(missing_block$End[i-1]+1, missing_block$Start[i]-1)}
  }
  
  ## Updating final variables from training block
  training_block$Gap = training_block$End - training_block$Start + 1
  training_block$Train = TRUE
  
  ## Combining missing and training blocks
  block = rbind(missing_block, training_block) %>% dplyr::arrange(Start)
  
  ## Specifying which data blocks are valid for training process
  block$Valid = ifelse(block$Train == FALSE, NA, 
                       ifelse(block$Gap >= (window + forecast), TRUE, FALSE))
  return(block)
}


set.seed(52)
X = interpTools::simXt(N = 100, numTrend = 1, numFreq = 2)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.1, g = 2, K = 1)[[1]]$p0.1$g2[[1]]

my_block = block_function(X0, window = 5, forecast = 1)

which(my_block$Valid == TRUE)

training_index = which(my_block$Valid == TRUE)
prediction_index = which(is.na(my_block$Valid))

## findMissingBlock function adapted from tsinterp's findBlocks function
##
## Takes data mask as input and returns block info.

findMissingBlock <- function(mask) {
  Nlen <- length(mask)
  mask <- which(is.na(mask))
  # case: there are missing points
  if(length(mask) > 0) {
    diffs <- mask[-1] - mask[-length(mask)]
    diffs <- which(diffs > 1)
    
    # case: 1 gap only, possibly no gaps
    if(length(diffs)==0) {
      blocks <- matrix(data=0, nrow=1, ncol=2)
      blocks[1, 1:2] <- c(mask[1], mask[length(mask)])
    } else {
      blocks <- matrix(data=0,nrow=length(mask),ncol=2)
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
  blocks = as.data.frame(blocks) %>% dplyr::rename(Start = V1, End = V2)
  blocks$Gap = blocks$End - blocks$Start + 1
  blocks$Train = FALSE
  return(blocks)
}









## ---------------------------------------------


blocking <- function(x, window, forecast){
  
  ## If necessary, returning error that entire series is NA
  if (sum(is.na(x)) == length(x)){print('Entire series is NA.'); return(NULL)}
  
  
  
  ## If necessary, removing missing values from beginning and end of series
  while(is.na(x[1])){x = x[-1]}
  while(is.na(x[length(x)])){x = x[-length(x)]}
  
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
  block$Valid = ifelse(block$Complete == 0, NA, 
                       ifelse(block$Gap >= (window + forecast), TRUE, FALSE))
  return(block)
}
## *****************************
## also want the function to return an updated series if x starts or end with NA values.
## *****************************


x = c(NA, NA, 1, 2, 3, 4, 5, 6, 7, 8, 
      NA, NA, NA, 10, 11, NA, NA, 2, 3, 4)
blocking(x, window = 2, forecast = 1)


x = c(1, 2, 3, 4, 5, 6, 7, 8, NA, NA, 
      NA, 10, 11, NA, NA, 2, 3, 4, NA, NA)
blocking(x, window = 2, forecast = 1)


x = c(1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 2, 3, 4)
blocking(x, window = 2, forecast = 1)

x = c(NA, NA, NA, NA, NA, NA, NA)
blocking(x, window = 2, forecast = 1)




x = c(1, 2, 3, 4, 5, NA, NA, NA, 9, 10, 11, NA, NA, 14, 15)
blocking(x, window = 2, forecast = 1)



set.seed(42)
X = interpTools::simXt(N = 100)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.3, g =5, K = 1)[[1]]$p0.3$g5[[1]]
X0

blocking(X0, window = 5, forecast = 1)

training_index = which(my_block$Valid == TRUE)
prediction_index = which(is.na(my_block$Valid))





  