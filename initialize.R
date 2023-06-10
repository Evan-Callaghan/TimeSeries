########################################
## Neural Network Time Series Imputer ##
########################################

## Initializer: All code related to initializing the imputation process


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)

## Notation Notes
## -----------------------

#' @param x0 {list}; List containing the original incomplete time series ("x naught")
#' @param xV {list}; List containing the current version of imputed series ("x version")

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
  xV = tsinterp::linInt(x0, blocks) ## Initial imputation using linear interpolation
  return(xV)
}