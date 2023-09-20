##################
## Testing File ##
##################


## Importing libraries
## -----------------------

library(dplyr)
library(tsinterp)
library(interpTools)
library(tensorflow)
library(keras)


## Configuring set-up (not always necessary)
## -----------------------

gpus = tf$config$experimental$list_physical_devices('GPU')
for (gpu in gpus){
  tf$config$experimental$set_memory_growth(gpu, TRUE)
}


## Importing functions
## -----------------------

source('APPROACHES/1_Autoencoder_updated.R')


## Simulations
## -----------------------

P = c(0.1, 0.2, 0.3)
G = c(5, 10, 25)
K = 3
METHODS = c('LOCF', 'LI', 'KAF')


## 1. Sunspots Data



set.seed(42)
X = interpTools::simXt(N = 1000)$Xt
plot_ts(X)

results = simulation_main(X, P, G, K, METHODS)

x0 = results[[1]]
xI = results[[2]]
performance = results[[3]]
aggregation = results[[4]]

simulation_plot(aggregation, criteria = 'RMSE', agg = 'mean', 
                title = 'Simulation Results:', levels = c('LOCF', 'LI', 'KAF'))


cleaned = simulation_cleaner(x0, xI, P, G, K, METHODS)


cleaned[[1]][1:3, 1:10]
cleaned[[2]][1:3, 1:11]

dim(cleaned[[1]])
dim(cleaned[[2]])

## output from x0: [[1]]$p0.3$g10[[3]]
## output from xI: [[1]]$NNI$p0.3$g10[[3]]






# Initializing data-frame to store results
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







## Simulations:
## ---------------------


P = c(0.1, 0.2, 0.3)
G = c(5, 10)
K = 3
METHODS = c('HWI', 'LI', 'NNI')

set.seed(42)
X = interpTools::simXt(N = 1000)$Xt
plot_ts(X)

results = simulation_main(X, P, G, K, METHODS)

x0 = results[[1]]
xI = results[[2]]
performance = results[[3]]
aggregation = results[[4]]

simulation_plot(aggregation, criteria = 'RMSE', agg = 'mean', 
                title = 'Simulation Results:', levels = c('HWI', 'LI', 'NNI'))









## Time Series:

## Create function:
## Scales the time series
## Return a plot
## Maybe remove titles from these.
## Uniform dimensions when exporting.
## Want them to be the same size in latex.

ts = read.csv('monthly-sunspots.csv')
ts = ts$Sunspots

X_t = data.frame(index = seq(1, length(ts)), value = ts)

ggplot(data = X_t, aes(x = index, y = value)) +
  geom_line(color = "#204176") + 
  geom_point(color = '#204176', size = 0.3) +
  labs(title = paste0('Time Series: Monthly Sunspot Data'), x = "Index", y = "Value") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0, face = 'bold', size = 18), 
        axis.text = element_text(color = 'black', size = 8), 
        axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
        axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
        panel.grid = element_line(color = 'grey', linewidth = 0.5, linetype = 'dotted'))


plot_ts(ts, 'test')






