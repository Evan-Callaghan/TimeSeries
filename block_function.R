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
  