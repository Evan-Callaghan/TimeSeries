###################################
## Simulation Plotting Functions ##
###################################


## Defining all functions
## -----------------------


#' plot_ts
#' 
#' Function to formalize the time series plotting process. Follows a standard form for which all time 
#' series plots/visualization will be displayed in the thesis.
#' @param x {list}; List object containing the time series to be plotted
#' @param title {string}; Title of the plot (default is empty)
#'
plot_ts <- function(x, title = ''){
  
  X_t = data.frame(index = seq(1, length(x)), value = x)
  
  plt = ggplot(data = X_t, aes(x = index, y = value)) +
    geom_line(color = "#476d9e", linewidth = 0.8) + 
    geom_point(color = '#476d9e', size = 0.4) +
    labs(title = paste0(title), x = "Index", y = "Value") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 14), 
          axis.text = element_text(color = 'black', size = 10), 
          axis.title.x = element_text(color = 'black', size = 14, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 14, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'grey70', linewidth = 0.5, linetype = 'dotted'),
          text=element_text(family="Helvetica"))
  return(plt)
}


#' simulation_plot
#' 
#' Function to formalize the imputation simulation plotting process. Follows a standard form for which 
#' all time series imputation plots/visualization will be displayed in the thesis.
#' @param data {dataframe}; Data frame containing complete simulation results.
#' @param metric {string}; Desired criteria for imputation performance evaluation
#' @param aggregation {string}; Function type for aggregating data across K iterations
#' @param methods {list}; List of all desired methods to be displayed
#' @param title {string}; Desired title for the returned plot
#'
simulation_plot <- function(data, metric, aggregation, methods, title = ''){
  
  # Performing the aggregation
  grouped_results = data %>% dplyr::select(Method, P, G, all_of(metric)) %>% 
    dplyr::filter(Method %in% methods) %>%
    dplyr::rename('value' = all_of(metric)) %>% 
    dplyr::group_by(Method, P, G) %>%
    dplyr::summarise(agg_value = eval(parse(text = paste0(aggregation, '(value, na.rm = TRUE)'))))
  
  # Creating colour palette
  colors = c("#91cff2", "#85bee4", "#78add5", "#6c9dc7", "#608cb9", "#537dab", "#476d9e", "#3b5e90", "#2e4f83", "#204176")
  col = colorRampPalette(colors = colors)(100)
  
  # Creating plot
  plt = ggplot(grouped_results, aes(as.factor(P), as.factor(G), fill = agg_value)) +
    geom_tile(color = '#204176', linewidth = 0.1) +
    facet_grid(~ factor(Method, levels = methods)) +
    labs(title = title, 
         subtitle = paste0('Metric: ', metric, '   Aggregation: ', aggregation),
         x = "Missing Proportion (P)", 
         y = "Gap Width (G)") +
    scale_fill_gradientn(colours = col, values = c(0,1)) + 
    guides(fill = guide_colourbar(label = TRUE, ticks = TRUE)) +
    geom_text(aes(label = round(agg_value, 3)), color = "white", size = 3.5) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 14), 
          plot.subtitle = element_text(hjust = 0.5, size = 12, face = 'italic'),
          strip.background = element_rect(fill = 'white'), 
          strip.text = element_text(color = 'black', face = 'bold', size = 10), 
          panel.spacing = unit(0.3, 'lines'), 
          axis.text = element_text(color = 'black', size = 10), 
          axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'white'), 
          legend.position = "none", 
          legend.key.height = unit(1, 'cm'), 
          legend.title.align = -0.5)
  return(plt)
}


#' error_distribution_facet
#' 
#' Function which creates a facet grid and displays the distribution of a specified
#' error metric by method, P, and G. 
#' @param data {dataframe}; Data frame containing complete simulation results.
#' @param metric {string}; Desired criteria for imputation performance evaluation
#' @param methods {list}; List of all desired methods to be displayed
#' @param title {string}; Desired title for the returned plot
#'
error_distribution_facet <- function(data, metric, methods, title = ''){
  
  # Defining method colors
  method_colors = c(HWI = "chocolate4", LI = "grey50", NNI = "dodgerblue3")
  
  # Cleaning the data-frame
  data_filtered = data %>% dplyr::filter(Method %in% methods) %>%
    dplyr::select(Method, P, G, all_of(metric)) %>%
    dplyr::rename('value' = all_of(metric))
  
  # Creating the plot
  plt = ggplot(data = data_filtered, aes(x = value, fill = Method, group = Method)) +
    facet_grid(rows = vars(factor(G, levels = c(50, 25, 10, 5))), cols = vars(factor(P))) +
    geom_histogram(alpha = 0.55, color = 'black', position = 'identity') + 
    geom_density(color = 'black', alpha = 0.6, adjust = 2) +
    labs(title = title, 
         subtitle = paste0('Metric: ', metric),
         x = 'Missing Proportion (P)', 
         y = 'Gap Width (G)', 
         fill = 'Method:') +
    theme_bw() +
    scale_fill_manual(values = method_colors) +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 14), 
          plot.subtitle = element_text(hjust = 0.5, size = 12, face = 'italic'),
          strip.background = element_rect(fill = 'grey75'), 
          strip.text = element_text(color = 'black', face = 'bold', size = 10),
          strip.text.y = element_text(angle = 0),
          panel.spacing = unit(0.3, 'lines'), 
          axis.text = element_text(color = 'black', size = 8), 
          axis.title.x = element_text(color = 'black', size = 12, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 12, margin = margin(r = 8)), 
          legend.box.background = element_rect(),
          legend.box.margin = margin(4, 4, 4, 4), 
          legend.position = "right", 
          legend.key.height = unit(1, 'cm'), 
          legend.title.align = -0.5, 
          legend.background = element_rect(fill = 'white'))
  return(plt)
}



#' error_crosssection_plot
#' 
error_crosssection_plot <- function(data, metric, methods, title = ''){
  
  data_filtered = data %>% dplyr::filter(Method %in% methods) %>%
    dplyr::rename('value' = all_of(metric)) %>%
    dplyr::select(Method, P, G, value) %>%
    dplyr::group_by(Method, P, G) %>%
    dplyr::summarise(lower = quantile(value, 0.05), 
                     median = quantile(value, 0.5), 
                     upper = quantile(value, 0.95))
  
  colors = c('gold', 'red', 'black')
  
  plt = ggplot()
  
  for (i in 1:length(methods)){
    
    data_temp = data_filtered %>% dplyr::filter(Method == methods[i])
    
    plt = plt +
      geom_ribbon(data = data_temp, aes(x = P, ymin = lower, ymax = upper), fill = colors[i], alpha = 0.2) +
      geom_line(data = data_temp, aes(x = P, y = median), color = colors[i]) +
      geom_point(data = data_temp, aes(x = P, y = median), color = colors[i]) +
      geom_errorbar(data = data_temp, aes(x = P, ymin = lower, ymax = upper), color = colors[i], width = 0.005) +
      facet_grid(rows = vars(factor(G, levels = c(50, 25, 10, 5))))
  }
  
  plt = plt + ylim(0.09, 0.25) +
    labs(title = title, x = 'Missing Proportion (P)', y = 'Gap Width (G)') +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 14), 
          strip.background = element_rect(fill = 'white'), 
          strip.text.y.right = element_text(color = 'black', face = 'bold', size = 10, angle = 0), 
          panel.spacing = unit(0.8, 'lines'), 
          axis.text = element_text(color = 'black', size = 12), 
          axis.title.x = element_text(color = 'black', size = 10, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 10, margin = margin(r = 8)), 
          axis.text.y = element_text(color = 'black', size = 8, margin = margin(r = 8)),
          legend.box.background = element_rect(),
          legend.box.margin = margin(4, 4, 4, 4), 
          legend.position = "right", 
          legend.key.height = unit(1, 'cm'), 
          legend.title.align = -0.5)
  return(plt)
}



