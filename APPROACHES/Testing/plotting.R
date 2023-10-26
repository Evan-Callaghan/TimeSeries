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
#' @param aggregation {aggregation_pf}; Aggregation of imputation performance across methods
#' @param criteria {string}; Desired criteria for imputation performance evaluation (default 'RMSE)
#' @param agg {string}; Function type for aggregating data across K iterations (default 'mean')
#' @param title {string}; Desired title for the returned plot
#' @param levels {numeric}; Vector containing the method names in the desired order for the plot
#'
simulation_plot <- function(data, criteria, aggregation, levels, title = ''){
  
  # Performing the aggregation
  grouped_results = data %>% dplyr::select(Method, P, G, all_of(criteria)) %>% 
    dplyr::rename('value' = all_of(criteria)) %>% 
    dplyr::group_by(Method, P, G) %>%
    dplyr::summarise(agg_value = eval(parse(text = paste0(aggregation, '(value, na.rm = TRUE)'))))
  
  
  # Creating colour palette
  colors = c("#91cff2", "#85bee4", "#78add5", "#6c9dc7", "#608cb9", "#537dab", "#476d9e", "#3b5e90", "#2e4f83", "#204176")
  col = colorRampPalette(colors = colors)(100)
  
  # Creating plot
  plt = ggplot(grouped_results, aes(as.factor(P), as.factor(G), fill = agg_value)) +
    geom_tile(color = '#204176', linewidth = 0.1) +
    facet_grid(~ factor(Method, levels = levels)) +
    labs(title = title, 
         x = "Missing Proportion (P)", 
         y = "Gap Width (G)", 
         fill = criteria) +
    scale_fill_gradientn(colours = col, values = c(0,1)) + 
    guides(fill = guide_colourbar(label = TRUE, ticks = TRUE, title = criteria)) +
    geom_text(aes(label = round(agg_value, 3)), color = "white", size = 4) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0, face = 'bold', size = 14), 
          strip.background = element_rect(fill = 'white'), 
          strip.text = element_text(color = 'black', face = 'bold', size = 12), 
          panel.spacing = unit(0.8, 'lines'), 
          axis.text = element_text(color = 'black', size = 12), 
          axis.title.x = element_text(color = 'black', size = 14, margin = margin(t = 8)), 
          axis.title.y = element_text(color = 'black', size = 14, margin = margin(r = 8)), 
          panel.grid = element_line(color = 'white'), 
          legend.box.background = element_rect(),
          legend.box.margin = margin(4, 4, 4, 4), 
          legend.position = "right", 
          legend.key.height = unit(1, 'cm'), 
          legend.title.align = -0.5)
  return(plt)
}