###########################
## Baseline Testing File ##
###########################

#* Requirements:
#*    tsinterp version 0.2.1
#*    interpTools version 0.1.0
#*    Three data sets (as csv files)
#*    
#* Instructions:
#*    Load all libraries with required versions. 
#*    Import all functions from the 'testing_functions' file.
#*    All areas with '...' are dependent on where the files are stored (adjust as needed).
#* 


## Importing libraries
## -----------------------

library(dplyr)
library(parallel)
library(tsinterp)
library(interpTools)


## Importing functions
## -----------------------

source('APPROACHES/Testing/baseline_helper.R')


## Simulation Parameters
## -----------------------

P = c(0.1, 0.2, 0.3, 0.4)
G = c(5, 10, 25, 50)
K = 50

METHODS = c('NN', 'LI', 'NCS', 'FMM', 'HCS', 'SI', 'KAF', 'KKSF', 'LOCF', 'NOCB', 'SMA', 
            'LWMA', 'EWMA', 'RMEA', 'RMED', 'RMOD', 'RRND', 'HWI')

numCores = 16


## Reading the data sets
## -----------------------

sunspots = read.csv('Data/Cleaned/sunspots.csv')$sunspots
apple = read.csv('Data/Cleaned/apple.csv')$apple
high_snr = read.csv('Data/Cleaned/simulated.csv')$high_snr
low_snr = read.csv('Data/Cleaned/simulated.csv')$low_snr
modulated = read.csv('Data/Cleaned/simulated.csv')$modulated


## Simulation 1: Sunspots Data
## -----------------------

## Running the imputation simulation
sunspots_sim = simulation_main(sunspots, P, G, K, METHODS, numCores)

## Exporting simulation performance as a csv file
write.csv(sunspots_sim, 'APPROACHES/SIMULATIONS/SimRound1_October2023_Baseline_sunspots.csv', row.names = FALSE)



## Simulation 2: Apple Stock Data
## -----------------------

## Running the imputation simulation
apple_sim = simulation_main(apple, P, G, K, METHODS, numCores)

## Exporting simulation performance as a csv file
write.csv(apple_sim, 'APPROACHES/SIMULATIONS/SimRound1_October2023_Baseline_apple.csv', row.names = FALSE)



## Simulation 3: High SNR Time Series
## -----------------------

## Running the imputation simulation
high_snr_sim = simulation_main(high_snr, P, G, K, METHODS, numCores)

## Exporting simulation performance as a csv file
write.csv(high_snr_sim, 'APPROACHES/SIMULATIONS/SimRound1_October2023_Baseline_high_snr.csv', row.names = FALSE)



## Simulation 4: Low SNR Time Series
## -----------------------

## Running the imputation simulation
low_snr_sim = simulation_main(low_snr, P, G, K, METHODS, numCores)

## Exporting simulation performance as a csv file
write.csv(low_snr_sim, 'APPROACHES/SIMULATIONS/SimRound1_October2023_Baseline_low_snr.csv', row.names = FALSE)



## Simulation 5: Frequency Modulated Time Series
## -----------------------

## Running the imputation simulation
modulated_sim = simulation_main(modulated, P, G, K, METHODS, numCores)

## Exporting simulation performance as a csv file
write.csv(modulated_sim, 'APPROACHES/SIMULATIONS/SimRound1_October2023_Baseline_modulated.csv', row.names = FALSE)

