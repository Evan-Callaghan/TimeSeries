########################################
## Neural Network Time Series Imputer ##
########################################

## TESTING FRAMEWORK: 


## Importing libraries
## -----------------------

library(tsinterp)
library(interpTools)
source('estimate.R')
source('simulate.R')
source('impute.R')
source('initialize.R')
source('main.R')
source('performance.R')


## Defining all functions
## -----------------------


#' FRAMEWORK
#' 
#' Function which facilitates the testing of the Neural Network Imputer (NNI) versus other methods implemented in the
#' interpTools package.
#' @param X {list}; List object containing a complete time series (should be scaled to (0,1))
#' @param P {list}; Proportion of missing data to be tested 
#' @param G {list}; Gap width of missing data to be tested 
#' @param K {integer}; Number of iterations for each P and G combination
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
FRAMEWORK <- function(X, P, G, K, METHODS){
  
  ## Saving length of experiment
  M = length(P) * length(G) * K * length(METHODS)
  
  ## Impose
  x0 = interpTools::simulateGaps(list(X), P, G, K)
  
  ## Impute
  xI = my_parInterpolate(x0, METHODS)
  
  ## Evaluate
  performance = my_performance(X, xI, x0)
  
  ## Aggregate
  aggregation = interpTools::aggregate_pf(performance)
  
  ## Return
  return(aggregation)
}

X = interpTools::simXt(N = 1000, mu = 0)$Xt
results = FRAMEWORK(X, P = c(0.1, 0.2, 0.3), G = c(10, 25, 50), K = 5, METHODS = c('LI', 'EWMA', 'LOCF'))
results


## Simple plot of aggregated RMSE across methods
interpTools::plotMetrics(results, p = 0.1, g = 10, metric = 'LCL')

## Producing 
multiHeatmap(crit = 'RMSE', agEval = results, m = c('LI', 'EWMA', 'LOCF'), by = 'method')

## 
# interpTools::multiSurface(metric = 'RMSE', agObject = results)



select_res = as.data.frame(results$D1$p0.1$g10$NNI) %>% 
  dplyr::select(gap_width, prop_missing, dataset, method, mean, iqr) %>%
  dplyr::filter(row.names(as.data.frame(results$D1$p0.1$g10$NNI)) %in% c('MAE', 'RMSE', 'LCL'))



class(interpTools::compileMatrix(results))




#' my_parInterpolate
#' 
#' Function which acts as a wrapper to the interpTools interpolation process and also adds the ability to use the Neural 
#' Network Imputer (NNI). 
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#' @param METHODS {list}; List of method to consider for imputation (supports 'NNI' and all others from interpTools)
#'
my_parInterpolate <- function(x0, METHODS){
  
  ## If the list contains NNI...
  if ('NNI' %in% METHODS){
    
    ## Removing NNI from methods list
    METHODS = METHODS[METHODS != 'NNI']
    
    ## Calling interpTools with remaining methods
    xI_all = interpTools::parInterpolate(x0, methods = METHODS)
    
    ## Performing NNI imputation
    xI_NNI = my_parInterpolate_NNI(x0) 
    
    ## Joining the two results
    xI = list(c(xI_all[[1]], xI_NNI[[1]]))
  }
  
  else {
    
    ## Otherwise, just perform imputation with interpTools
    xI = interpTools::parInterpolate(x0, methods = METHODS)
  }
  
  ## Returning the imputed series
  return(xI)
}



#' my_parInterpolate_NNI
#' 
#' Function which acts as a wrapper to the NNI interpolation process. Takes an incomplete time series as input and uses NNI
#' to produce the interpolated series.
#' @param x0 {list}; List object containing an incomplete time series (should be scaled to (0,1))
#'
my_parInterpolate_NNI <- function(x0){
  
  ## Defining helpful variables
  D = 1
  M = 1
  P = length(x0[[1]])
  G = length(x0[[1]][[1]])
  K = length(x0[[1]][[1]][[1]])
  numCores = detectCores()
  
  ## Initializing lists to store interpolated series
  int_series = lapply(int_series <- vector(mode = 'list', M), function(x)
    lapply(int_series <- vector(mode = 'list', P), function(x) 
      lapply(int_series <- vector(mode = 'list', G), function(x) 
        x <- vector(mode = 'list', K))))
  
  int_data = list()
  
  ## Setting up the function call
  function_call = paste0("run_simulation(x, 10, 300, 'noise', 0.05, pi/6, 1)")
  
  ## Performing imputation
  int_series[[M]] = lapply(x0[[D]], function(x){
    lapply(x, function(x){
      lapply(x, function(x){
        eval(parse(text = function_call))})}
    )})
  
  ## Applying the function name 
  names(int_series) = c('NNI')
  
  ## Saving the imputed series
  int_data[[D]] = int_series
  
  ## Returning the imputed series
  return(int_data)
}





###################################################
###################################################
###################################################
## TEMP: Code copied from interpTools for some minor upgrades





my_new_multiHeatmap <- function(agg, P, G, METHODS, crit = 'MAE', f = 'median', 
                                colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")){
  
  ## Initializing a data-frame 
  data = data.frame()
  
  ## Adding data from all aggregations
  for (p in P){
    for (g in G){
      for (method in METHODS){
        temp = eval(parse(text = paste0('as.data.frame(agg$D1$p', p, '$g', g, '$', method, ')')))
        temp$metric = rownames(temp); rownames(temp) = NULL
        data = rbind(data, temp)
      }
    }
  }
  
  selecting = paste0('data %>% dplyr::select(method, prop_missing, gap_width, metric, ', f, ')')
  
  data = eval(parse(text = selecting)) %>% 
    dplyr::filter(metric == crit)
  
  return(data)
}

df = my_new_multiHeatmap(results, P = c(0.1, 0.2, 0.3), G = c(10, 25, 50), METHODS = c('LI', 'EWMA', 'LOCF'))







crit = 'RMSE'
f = 'median'
to_select = paste0('method, prop_missing, gap_width, ', f)

df %>% dplyr::select(to_select) %>%
  dplyr::filter(row.names(df) == crit)



to_select = 'dplyr::select(Second)'
df = data.frame(First = c(1, 2, 4), Second = c(1, 2, 3))

df %>% eval(parse(text = to_select))



P = c(0.1, 0.2, 0.3)
G = c(10, 25, 50)
METHODS = c('LI', 'EWMA', 'LOCF')
df = data.frame()

for (p in P){
  for (g in G){
    for (method in METHODS){
      df = rbind(df, eval(parse(text = paste0('as.data.frame(results$D1$p', p, '$g', g, '$', method, ')'))))
    }
  }
}

df




m = c('LI', 'EWMA', 'LOCF')

M <- length(m)
C <- 1
D <- 1
d <- 1
colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")
crit = 'RMSE'

z_list <- interpTools::compileMatrix(results)[['median']]

# get legend
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

heatmapList <- list()

attr <- list()


bound = M
by_vec = m 
rng <- range(z_list[['RMSE']][m])

for(cr in 1:bound){
  
  if(cr == 1){
    titles <- paste0("'Dataset ",d,"'")
    titles_theme <- element_text(hjust = 0.5)
    axislabels.x <- element_blank()
    axislabels.y <- element_blank()
    axistext.y <- element_text()
    axisticks <- element_blank()
  }
  else if(cr == bound){
    titles <- rep("",D)
    axislabels.x <- element_text()
    axislabels.y <- element_text()
    axisticks <- element_blank()
    axistext.y <- element_blank()
    titles_theme <- element_blank()
  }
  else{
    titles <- rep("",D)
    titles_theme <- element_blank()
    axislabels.x <- element_blank()
    axislabels.y <- element_blank()
    axistext.y <- element_blank()
    axisticks <- element_blank()
  }
  
  plott <- list()
  
  rownames(z_list[[crit]][[m[cr]]][[d[1]]]) <- round(as.numeric(gsub("p","",rownames(z_list[[crit]][[m[cr]]][[d[1]]]), fixed = TRUE)),2)
  colnames(z_list[[crit]][[m[cr]]][[d[1]]]) <- round(as.numeric(gsub("g","",colnames(z_list[[crit]][[m[cr]]][[d[1]]]), fixed = TRUE)),2)
  
  plott[[1]] <- melt(z_list[[crit]][[m[cr]]][[d[1]]])
  colnames(plott[[1]]) <- c("p","g", "value")
  
  rng = range(z_list[['RMSE']][m])
  col = colorRampPalette(colors = colors)(100)
  
  plotList <- list()
  
  
  
  plotList[[1]] <- ggplot(plott[[1]], aes(as.factor(p), as.factor(g), fill = value)) + 
    geom_tile() + 
    scale_fill_gradientn(colours = col, values = c(0,1), limits = rng) +
    
    labs(x = "Missing Proportion (P)", y = "Gap Width (G)", fill = by_vec[cr], title = eval(parse(text = titles[1]))) + 
    theme_minimal() + 
    
    theme(legend.position = "none",
          
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = axislabels.x,
          axis.text.y = element_text(),
          plot.title = titles_theme,
          axis.ticks.x = axisticks,
          axis.ticks.y = axisticks)
  
  # making dummy plot to retrieve legend
  
  dumPlot <- ggplot(plott[[1]], aes(as.factor(p), as.factor(g), fill = value)) + 
    geom_tile() + 
    scale_fill_gradientn(colours = col, values = c(0,1), limits = rng) +
    theme(legend.position = "bottom", legend.direction = "vertical") + 
    labs(fill = paste0('RMSE'," (",'median',")"), x = "", y = "")
  
  myLegend <- g_legend(dumPlot)
  
  
  h_string <- paste0("grid.arrange(plotList[[1]], ncol = ",D,", widths = c(10), right = m[",cr,"])", collapse = "")
  
  heatmapList[[cr]] <- eval(parse(text = h_string))
}

names(heatmapList) <- by_vec
call <- paste0("heatmapList[[",1:(bound-1),"]],")
call <- c("grid.arrange(",call,paste0("heatmapList[[",bound,"]], nrow = ",1,", bottom = 'proportion missing', left = 'gap width')"))

plotWindow <- eval(parse(text = call))
plotWindow <- grid.arrange(plotWindow, myLegend, ncol = 3, heights = 10, widths = c(20, 20, 20))





length(results[[1]])




length(results)

as.data.frame(results[]$[]$[]) 
as.data.frame(results[]$[]$[]) 
as.data.frame(results[]$[]$[]) 

df = rbind(...)


crit = 'RMSE'
f = 'median'
to_select = paste0('method, prop_missing, gap_width, ', f)

df %>% dplyr::select(to_select) %>%
  dplyr::filter(row.names(df) == crit)



ggplot(imp_mae, aes(P, G, fill = mae)) +
  geom_tile(color = "gray95",lwd = 0.5, linetype = 1) +
  facet_grid(~ Method) +
  labs(title = 'Simulation Results with N = 1000 (Avg. MAE)', 
       x = "Missing Proportion (P)", 
       y = "Gap Width (G)", 
       fill = 'MAE') +
  scale_fill_gradientn(colours = col, values = c(0,1)) + 
  guides(fill = guide_colourbar(label = FALSE, ticks = FALSE)) +
  geom_text(aes(label = round(mae, 3)), color = "white", size = 3) +
  theme_minimal()











## This new function is going to plot the performance of different methods as a heat map for a given
## metric and "f" argument (mean. median. IQR, etc...). Making changes from Sophie's multiHeatmap function
## because we are not considering 'by = crit' we are only comparing across methods.
my_new_multiHeatmap <- function(crit, agEval, m, f = "median", 
                                colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")){
  
  ## LOGICAL CHECKS ############
  
  #if(sum(duplicated(d) != 0)) stop(paste0("'d' contains redundant elements at position(s): ", paste0(c(1:length(d))[duplicated(d)], collapse = ", ") ))
  if(sum(duplicated(m) != 0)) stop(paste0("'m' contains redundant elements at position(s): ", paste0(c(1:length(m))[duplicated(m)], collapse = ", ") ))
  if(sum(duplicated(crit) != 0)) stop(paste0("'crit' contains redundant elements at position(s): ", paste0(c(1:length(crit))[duplicated(crit)], collapse = ", ") ))
  
  #if(by != "crit" & by != "method") stop("'by' must be either 'crit' or 'method'.")
  #if(by == "crit" & length(crit) < 2) stop("Only one criterion was chosen. Please specify at least one more, or use 'heatmapGrid()' instead.")
  #if(by == "method" & length(m) < 2) stop("Only one method was chosen. Please specify at least one more, or use 'heatmapGrid()' instead.")
  
  #if(class(agEval) != "agEvaluate") stop("'agEval' object must be of class 'agEvaluate'. Please use agEvaluate().")
  if(class(agEval) != "aggregate_pf") stop("'agEval' object must be of class 'aggregate_pf'. Please use agEvaluate().")
  
  
  #f(by == "method" & length(crit) != 1) stop("'crit' must contain only a single character element if you wish to arrange by method.")
  if(length(crit) != 1) stop("'crit' must contain only a single character element if you wish to arrange by method.")
  
  if(length(f) != 1) stop("'f' must contain only a single character element.")
  #if(length(by) != 1) stop("'by' must contain only a single character element.")
  
  #if(by == "crit" & length(m) != 1) stop("'m' must contain only a single character element if you wish to arrange by criterion.")
  
  if(!all(m %in%  names(agEval[[1]][[1]][[1]]))) stop(paste0("Method(s) '", paste0(m[!m%in% names(agEval[[1]][[1]][[1]])], collapse = ", "), "' not found. Possible choices are: '", paste0(names(agEval[[1]][[1]][[1]]), collapse = "', '"),"'."))
  #if(!all(paste0("D",d) %in% names(agEval))) stop("Dataset(s) ", paste0(d[!paste0("D",d) %in% names(agEval)], collapse = ", ")," not found. Possible choices are: ", paste0(gsub("D", "",names(agEval)), collapse = ", "))
  if(!all(f %in% names(agEval[[1]][[1]][[1]][[1]]))) stop(paste0(c("f must be one of: '",paste0(names(agEval[[1]][[1]][[1]][[1]]), collapse = "', '"),"'."), collapse = ""))
  if(!all(crit %in% rownames(agEval[[1]][[1]][[1]][[1]]))) stop(paste0("Criterion '",crit,"' must be one of ", paste(rownames(agEval[[1]][[1]][[1]][[1]]),collapse = ", "),"."))
  
  if(length(colors) <2) stop("'colors' must contain at least two colors (each in HTML format: '#xxxxxx')")
  
  ##################
  
  M <- length(m)
  C <- 1
  D <- 1
  
  z_list <- compileMatrix(agEval)[[f]]
  
  # get legend
  g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
  
  heatmapList <- list()
  
  attr <- list()
  
  
  bound = M
  by_vec = m 
  rng <- range(z_list[[crit]][m])
  
  for(cr in 1:bound){
    
    if(cr == 1){
      titles <- paste0("'Dataset ",d,"'")
      titles_theme <- element_text(hjust = 0.5)
      axislabels.x <- element_blank()
      axislabels.y <- element_blank()
      axistext.y <- element_text()
      axisticks <- element_blank()
    }
    else if(cr == bound){
      titles <- rep("",D)
      axislabels.x <- element_text()
      axislabels.y <- element_text()
      axisticks <- element_blank()
      axistext.y <- element_blank()
      titles_theme <- element_blank()
    }
    else{
      titles <- rep("",D)
      titles_theme <- element_blank()
      axislabels.x <- element_blank()
      axislabels.y <- element_blank()
      axistext.y <- element_blank()
      axisticks <- element_blank()
    }
    
    plott <- list()
    
    for(vd in 1:D){
      rownames(z_list[[crit]][[m[cr]]][[d[vd]]]) <- round(as.numeric(gsub("p","",rownames(z_list[[crit]][[m[cr]]][[d[vd]]]), fixed = TRUE)),2)
      colnames(z_list[[crit]][[m[cr]]][[d[vd]]]) <- round(as.numeric(gsub("g","",colnames(z_list[[crit]][[m[cr]]][[d[vd]]]), fixed = TRUE)),2)
      
      plott[[vd]] <- melt(z_list[[crit]][[m[cr]]][[d[vd]]])
      colnames(plott[[vd]]) <- c("p","g", "value")
    }
    
    rng = range(z_list[[crit]][m])
    col = colorRampPalette(colors = colors)(100)
    
    plotList <- list()
    
    for(vd in 1:D){
      
      if(vd == 1){
        plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
          scale_fill_gradientn(colours = col, values = c(0,1),
                               limits = rng) +
          
          labs(x = "Missing Proportion (P)", y = "Gap Width (G)", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
          theme_minimal() + 
          
          theme(legend.position = "none",
                
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.text.x = axislabels.x,
                axis.text.y = element_text(),
                plot.title = titles_theme,
                axis.ticks.x = axisticks,
                axis.ticks.y = axisticks)
      }
      
      else if(vd != 1){
        plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
          scale_fill_gradientn(colours = col, values = c(0,1),
                               limits = rng) +
          
          labs(x = "proportion missing", y = "gap width", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
          theme_minimal() + 
          
          theme(legend.position = "none",
                
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.text.x = axislabels.x,
                axis.text.y = element_blank(),
                plot.title = titles_theme,
                axis.ticks.x = axisticks,
                axis.ticks.y = axisticks)
      }
      
    }
    
    # making dummy plot to retrieve legend
    
    dumPlot <- ggplot(plott[[1]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
      scale_fill_gradientn(colours = col, values = c(0,1),
                           limits = rng) +
      theme(legend.position = "bottom", legend.direction = "vertical") + 
      labs(fill = paste0(crit," (",f,")"), x = "", y = "")
    
    myLegend <- g_legend(dumPlot)
    
    if(D >= 3){
      h_string <- paste0("plotList[[",2:(D-1),"]],", collapse = "")
      rep_string <- c(paste0(rep("10,",D-1), collapse = ""),"10")
      rep_string <- paste0(rep_string, collapse = "")
      
      h_string <- c("grid.arrange(plotList[[1]],",h_string,paste0("plotList[[",D,"]], ncol = ",D,", widths = c(",rep_string,"), right = m[",cr,"])", collapse = ""))
    }
    
    else if(D == 2){
      h_string <- paste0("grid.arrange(plotList[[1]], plotList[[2]], ncol = ",D,", widths = c(10,10), right = m[",cr,"])", collapse = "")
    }
    
    else if(D == 1){
      h_string <- paste0("grid.arrange(plotList[[1]], ncol = ",D,", widths = c(10), right = m[",cr,"])", collapse = "")
    }
    
    heatmapList[[cr]] <- eval(parse(text = h_string))
  }
  
  names(heatmapList) <- by_vec
  
  if(bound > 1){
    call <- paste0("heatmapList[[",1:(bound-1),"]],")
  }
  else if(bound == 1){
    call = ""
  }
  call <- c("grid.arrange(",call,paste0("heatmapList[[",bound,"]], nrow = ",bound,", bottom = 'proportion missing', left = 'gap width')"))
  
  plotWindow <- eval(parse(text = call))
  plotWindow <- grid.arrange(plotWindow, myLegend, ncol = 1, heights = c(D*10, 10))
  
  
  
  return(plotWindow) 
}
























multiHeatmap <- function(crit, 
                         agEval, 
                         m, 
                         by = "crit", 
                         f = "median", 
                         d = 1:length(agEval), 
                         colors = c("#F9E0AA","#F7C65B","#FAAF08","#FA812F","#FA4032","#F92111")){
  
  ## LOGICAL CHECKS ############
  
  if(sum(duplicated(d) != 0)) stop(paste0("'d' contains redundant elements at position(s): ", paste0(c(1:length(d))[duplicated(d)], collapse = ", ") ))
  if(sum(duplicated(m) != 0)) stop(paste0("'m' contains redundant elements at position(s): ", paste0(c(1:length(m))[duplicated(m)], collapse = ", ") ))
  if(sum(duplicated(crit) != 0)) stop(paste0("'crit' contains redundant elements at position(s): ", paste0(c(1:length(crit))[duplicated(crit)], collapse = ", ") ))
  
  if(by != "crit" & by != "method") stop("'by' must be either 'crit' or 'method'.")
  if(by == "crit" & length(crit) < 2) stop("Only one criterion was chosen. Please specify at least one more, or use 'heatmapGrid()' instead.")
  if(by == "method" & length(m) < 2) stop("Only one method was chosen. Please specify at least one more, or use 'heatmapGrid()' instead.")
  
  #if(class(agEval) != "agEvaluate") stop("'agEval' object must be of class 'agEvaluate'. Please use agEvaluate().")
  if(class(agEval) != "aggregate_pf") stop("'agEval' object must be of class 'aggregate_pf'. Please use agEvaluate().")
  
  
  if(by == "method" & length(crit) != 1) stop("'crit' must contain only a single character element if you wish to arrange by method.")
  
  if(length(f) != 1) stop("'f' must contain only a single character element.")
  if(length(by) != 1) stop("'by' must contain only a single character element.")
  
  if(by == "crit" & length(m) != 1) stop("'m' must contain only a single character element if you wish to arrange by criterion.")
  
  if(!all(m %in%  names(agEval[[1]][[1]][[1]]))) stop(paste0("Method(s) '", paste0(m[!m%in% names(agEval[[1]][[1]][[1]])], collapse = ", "), "' not found. Possible choices are: '", paste0(names(agEval[[1]][[1]][[1]]), collapse = "', '"),"'."))
  if(!all(paste0("D",d) %in% names(agEval))) stop("Dataset(s) ", paste0(d[!paste0("D",d) %in% names(agEval)], collapse = ", ")," not found. Possible choices are: ", paste0(gsub("D", "",names(agEval)), collapse = ", "))
  if(!all(f %in% names(agEval[[1]][[1]][[1]][[1]]))) stop(paste0(c("f must be one of: '",paste0(names(agEval[[1]][[1]][[1]][[1]]), collapse = "', '"),"'."), collapse = ""))
  if(!all(crit %in% rownames(agEval[[1]][[1]][[1]][[1]]))) stop(paste0("Criterion '",crit,"' must be one of ", paste(rownames(agEval[[1]][[1]][[1]][[1]]),collapse = ", "),"."))
  
  if(length(colors) <2) stop("'colors' must contain at least two colors (each in HTML format: '#xxxxxx')")
  
  ##################
  
  M <- length(m)
  C <- length(crit)
  D <- length(d)
  
  z_list <- compileMatrix(agEval)[[f]]
  
  # get legend
  g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
  
  heatmapList <- list()
  
  attr <- list()
  
  if(by == "crit"){
    bound = C
    by_vec = crit
    
    for(cr in 1:bound){
      
      if(cr == 1){
        titles <- paste0("'Dataset ",d,"'")
        titles_theme <- element_text(hjust = 0.5)
        axislabels.x <- element_blank()
        axislabels.y <- element_blank()
        axistext.y <- element_text()
        axisticks <- element_blank()
        
      }
      else if(cr == bound){
        titles <- rep("",D)
        axislabels.x <- element_text()
        axistext.y <- element_blank()
        axislabels.y <- element_blank()
        axisticks <- element_blank()
        titles_theme <- element_blank()
      }
      else{
        titles <- rep("",D)
        titles_theme <- element_blank()
        axislabels.x <- element_blank()
        axislabels.y <- element_blank()
        axistext.y <- element_blank()
        axisticks <- element_blank()
      }
      
      plott <- list()
      
      for(vd in 1:D){
        rownames(z_list[[crit[cr]]][[m]][[d[vd]]]) <- round(as.numeric(gsub("p","",rownames(z_list[[crit[cr]]][[m]][[d[vd]]]), fixed = TRUE)),2)
        colnames(z_list[[crit[cr]]][[m]][[d[vd]]]) <- round(as.numeric(gsub("g","",colnames(z_list[[crit[cr]]][[m]][[d[vd]]]), fixed = TRUE)),2)
        
        
        plott[[vd]] <- melt(z_list[[crit[cr]]][[m]][[d[vd]]])
        colnames(plott[[vd]]) <- c("p","g", "value")
      }
      
      if(D >= 3){
        r_string <- paste0("plott[[",2:(D-1),"]]$value, ", collapse = "")
        r_string <- c("range(c(plott[[1]]$value,", r_string, paste0("plott[[",D,"]]$value))", collapse = ""))
      }
      
      else if(D == 2){
        r_string <- "range(c(plott[[1]]$value, plott[[2]]$value))"
      }
      
      else if(D == 1){
        r_string <- "range(plott[[1]]$value)"
      }
      
      rng = eval(parse(text = r_string))
      
      col = colorRampPalette(colors = colors)(100)
      
      plotList <- list()
      
      for(vd in 1:D){
        
        if(vd == 1){
          plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
            scale_fill_gradientn(colours = col, values = c(0,1),
                                 limits = rng) +
            
            labs(x = "proportion missing", y = "gap width", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
            theme_minimal() + 
            
            theme(legend.position = "none",
                  
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank(),
                  axis.text.x = axislabels.x,
                  axis.text.y = element_text(),
                  plot.title = titles_theme,
                  axis.ticks.x = axisticks,
                  axis.ticks.y = axisticks)
        }
        
        if(vd != 1){
          plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
            scale_fill_gradientn(colours = col, values = c(0,1),
                                 limits = rng) +
            
            labs(x = "proportion missing", y = "gap width", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
            theme_minimal() + 
            
            theme(legend.position = "none",
                  
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank(),
                  axis.text.x = axislabels.x,
                  axis.text.y = element_blank(),
                  plot.title = titles_theme,
                  axis.ticks.x = axisticks,
                  axis.ticks.y = axisticks)
        }
        
      }
      
      # make dummy plot to retrieve legend
      
      dumPlot <- ggplot(plott[[1]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
        scale_fill_gradientn(colours = col, values = c(0,1),
                             limits = rng) +
        theme(legend.position = "right") + 
        labs(fill = paste0(crit[cr]," (",f,")"), x = "", y = "")
      
      myLegend <-g_legend(dumPlot)
      
      if(D >= 3){
        h_string <- paste0("plotList[[",2:(D-1),"]],", collapse = "")
        rep_string <- c(paste0(rep("10,",D-1), collapse = ""),"10")
        rep_string <- paste0(rep_string, collapse = "")
        
        h_string <- c("grid.arrange(plotList[[1]],",h_string,paste0("plotList[[",D,"]], ncol = ",D,", widths = c(",rep_string,"))", collapse = ""))
      }
      
      else if(D == 2){
        h_string <- paste0("grid.arrange(plotList[[1]], plotList[[2]], ncol = ",D,", widths = c(10,10))", collapse = "")
      }
      
      else if(D == 1){
        h_string <- paste0("grid.arrange(plotList[[1]], ncol = ",D,", widths = c(10))", collapse = "")
      }
      
      myHeatmaps <- eval(parse(text = h_string))
      
      heatmapList[[cr]] <- grid.arrange(myHeatmaps, myLegend, ncol = 2, widths = c(40,5))
    }
    
    names(heatmapList) <- by_vec
    
    if(bound > 1){
      call <- paste0("heatmapList[[",1:(bound-1),"]],")
    }
    
    else if(bound == 1){
      call = ""
    }
    call <- c("grid.arrange(",call,paste0("heatmapList[[",bound,"]], nrow = ",bound,", bottom = 'proportion missing', left = 'gap width', top = m)"))
    
    plotWindow <- eval(parse(text = call))
  }
  
  else if(by == "method"){
    bound = M
    by_vec = m 
    rng <- range(z_list[[crit]][m])
    
    for(cr in 1:bound){
      
      if(cr == 1){
        titles <- paste0("'Dataset ",d,"'")
        titles_theme <- element_text(hjust = 0.5)
        axislabels.x <- element_blank()
        axislabels.y <- element_blank()
        axistext.y <- element_text()
        axisticks <- element_blank()
      }
      else if(cr == bound){
        titles <- rep("",D)
        axislabels.x <- element_text()
        axislabels.y <- element_text()
        axisticks <- element_blank()
        axistext.y <- element_blank()
        titles_theme <- element_blank()
      }
      else{
        titles <- rep("",D)
        titles_theme <- element_blank()
        axislabels.x <- element_blank()
        axislabels.y <- element_blank()
        axistext.y <- element_blank()
        axisticks <- element_blank()
      }
      
      plott <- list()
      
      for(vd in 1:D){
        rownames(z_list[[crit]][[m[cr]]][[d[vd]]]) <- round(as.numeric(gsub("p","",rownames(z_list[[crit]][[m[cr]]][[d[vd]]]), fixed = TRUE)),2)
        colnames(z_list[[crit]][[m[cr]]][[d[vd]]]) <- round(as.numeric(gsub("g","",colnames(z_list[[crit]][[m[cr]]][[d[vd]]]), fixed = TRUE)),2)
        
        plott[[vd]] <- melt(z_list[[crit]][[m[cr]]][[d[vd]]])
        colnames(plott[[vd]]) <- c("p","g", "value")
      }
      
      rng = range(z_list[[crit]][m])
      col = colorRampPalette(colors = colors)(100)
      
      plotList <- list()
      
      for(vd in 1:D){
        
        if(vd == 1){
          plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
            scale_fill_gradientn(colours = col, values = c(0,1),
                                 limits = rng) +
            
            labs(x = "Missing Proportion (P)", y = "Gap Width (G)", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
            theme_minimal() + 
            
            theme(legend.position = "none",
                  
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank(),
                  axis.text.x = axislabels.x,
                  axis.text.y = element_text(),
                  plot.title = titles_theme,
                  axis.ticks.x = axisticks,
                  axis.ticks.y = axisticks)
        }
        
        else if(vd != 1){
          plotList[[vd]] <- ggplot(plott[[vd]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
            scale_fill_gradientn(colours = col, values = c(0,1),
                                 limits = rng) +
            
            labs(x = "proportion missing", y = "gap width", fill = by_vec[cr], title = eval(parse(text = titles[vd]))) + 
            theme_minimal() + 
            
            theme(legend.position = "none",
                  
                  axis.title.x = element_blank(),
                  axis.title.y = element_blank(),
                  axis.text.x = axislabels.x,
                  axis.text.y = element_blank(),
                  plot.title = titles_theme,
                  axis.ticks.x = axisticks,
                  axis.ticks.y = axisticks)
        }
        
      }
      
      # making dummy plot to retrieve legend
      
      dumPlot <- ggplot(plott[[1]], aes(as.factor(p), as.factor(g), fill = value)) + geom_tile() + 
        scale_fill_gradientn(colours = col, values = c(0,1),
                             limits = rng) +
        theme(legend.position = "bottom", legend.direction = "vertical") + 
        labs(fill = paste0(crit," (",f,")"), x = "", y = "")
      
      myLegend <- g_legend(dumPlot)
      
      if(D >= 3){
        h_string <- paste0("plotList[[",2:(D-1),"]],", collapse = "")
        rep_string <- c(paste0(rep("10,",D-1), collapse = ""),"10")
        rep_string <- paste0(rep_string, collapse = "")
        
        h_string <- c("grid.arrange(plotList[[1]],",h_string,paste0("plotList[[",D,"]], ncol = ",D,", widths = c(",rep_string,"), right = m[",cr,"])", collapse = ""))
      }
      
      else if(D == 2){
        h_string <- paste0("grid.arrange(plotList[[1]], plotList[[2]], ncol = ",D,", widths = c(10,10), right = m[",cr,"])", collapse = "")
      }
      
      else if(D == 1){
        h_string <- paste0("grid.arrange(plotList[[1]], ncol = ",D,", widths = c(10), right = m[",cr,"])", collapse = "")
      }
      
      heatmapList[[cr]] <- eval(parse(text = h_string))
    }
    
    names(heatmapList) <- by_vec
    
    if(bound > 1){
      call <- paste0("heatmapList[[",1:(bound-1),"]],")
    }
    else if(bound == 1){
      call = ""
    }
    call <- c("grid.arrange(",call,paste0("heatmapList[[",bound,"]], nrow = ",bound,", bottom = 'proportion missing', left = 'gap width')"))
    
    plotWindow <- eval(parse(text = call))
    plotWindow <- grid.arrange(plotWindow, myLegend, ncol = 1, heights = c(D*10, 10))
  }
  
  return(plotWindow) 
}
