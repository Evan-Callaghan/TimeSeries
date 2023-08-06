## Blocking function



library(dplyr)

block_function <- function(x, window, forecast){
  
}


set.seed(52)
X = interpTools::simXt(N = 1000, numTrend = 1, numFreq = 2)$Xt
X0 = interpTools::simulateGaps(list(X), p = 0.1, g = 10, K = 1)[[1]]$p0.1$g10[[1]]



N = length(X0)
mask = ifelse(is.na(X0), TRUE, FALSE)

valid_index = which(mask == FALSE)
invalid_index = which(mask == TRUE)

valid_diff = which(valid_index[-1] - valid_index[-length(valid_index)] > 1)
invalid_diff = which(invalid_index[-1] - invalid_index[-length(invalid_index)] > 1)





check = mask[1]




storage = c()
for (i in 1:N){
  
  if (!is.na(i)){
    storage = c(storage, i)
  }
  
}


!(1 %in% c(1, 2, 3))


mask = which(is.na(X0))




N = length(X0)
diffs = which(mask[-1] - mask[-length(mask)] > 1)


# case: 1 gap only, possibly no gaps
if (length(diffs)==0){
  blocks <- matrix(data = 0, nrow = 1, ncol = 3)
  blocks[1, 1:2] <- c(mask[1], mask[length(mask)])
}
else{
  blocks <- matrix(data = 0, nrow = length(mask), ncol = 5)
  blocks[1,1] = 1
  blocks[1,2] = mask[1] - 1
  
  blocks[2, 1] <- mask[1]
  blocks[2, 2] <- mask[diffs[1]]
  k = 2
  for (j in 1:length(diffs)){
    k = k + 1
    blocks[k, 1:2] <- c(mask[diffs[j]+1], mask[diffs[j+1]])
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
  
  
  
  
  
  as.data.frame(blocks) %>% dplyr::rename(Start = V1, End = V2, Gap = V3)
  