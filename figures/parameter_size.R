# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)  # For combining individual plots

# Load your dataset (ensure this path is correct for your data file)
data <- read.csv("data/all_results_base_formatted.csv")


data_long <- data %>%
  pivot_longer(cols = starts_with("img_"),
               names_to = "label_type",
               values_to = "accuracy") %>%
  mutate(label_type = gsub("img_|label_", "", label_type)) %>%
  separate(label_type, into = c("image_type", "label_type"), sep = "\\.") %>%
  mutate(
    image_type = factor(image_type, levels = c("old", "new")),
    label_type = factor(label_type, levels = c("old", "new")),
    model = ifelse(finetuned == 1, "Finetuned", "Base"),
    model_type = model_path  # Ensure model_type is available for filtering
  )

# Now filter for the specific combinations you care about
data_long <- data_long %>%
  filter(
    (model == "Base" & image_type == "old" & label_type == "old") |
      (model == "Base" & image_type == "new" & label_type == "new")
  )

# Create a new column for grouping by combination of dataset and type
data_long$dataset_type <- interaction(data_long$dataset, data_long$type)

data_long <- data_long %>%
  mutate(dataset_type = interaction(dataset, type)) %>%
  mutate(dataset_type = factor(dataset_type, levels = c(
    "WebQA.feat",
    "VQA.counter",
    "OKVQA.counter",
    "WebQA.counter",
    "WebQA.conf"
  )))  # Set custom order for x-axis


# Filter data for baseline models only
data_baseline <- data_long %>%
  filter(model == "Base") %>%
  filter(label_type %in% c("old", "new"))

# Adjust the model_label_combination factor to only include Base categories
data_baseline$model_label_combination <- factor(paste(data_baseline$model, data_baseline$label_type, sep = "."), 
                                                levels = c("Base.old", "Base.new"))

# Update unique model types for the baseline plot
model_types_baseline <- unique(data_baseline$model_type)

# Function to create individual plots with only baseline models
create_baseline_plot <- function(data, show_y_axis = TRUE, show_legend = FALSE) {
  p <- ggplot(data, aes(x = dataset_type, y = accuracy, fill = model_label_combination)) +
    geom_bar(stat = "identity", position = "dodge", color = "black") +
    scale_fill_manual(values = c("Base.old" = "powderblue", 
                                 "Base.new" = "royalblue")) +
    labs(x = NULL, y = if (show_y_axis) "Accuracy Score" else NULL, fill = if (show_legend) "Model.Label" else NULL) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal(base_size = 18) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust=1, size=17),
          axis.title.x = element_blank(),
          axis.title.y = if (show_y_axis) element_text(size = 20) else element_blank(),
          axis.text.y = if (show_y_axis) element_text(size = 15) else element_blank(),
          axis.ticks.y = if (show_y_axis) element_line(color = "black") else element_blank(),
          panel.grid.major = element_line(color = "gray", size = 0.5),
          panel.grid.minor = element_line(color = "gray", size = 0.25),
          strip.text = element_text(size = 20),
          panel.spacing.x = unit(1, "lines"),
          legend.title = element_text(size = 15),  # Increase legend title size
          legend.text = element_text(size = 15),   # Increase legend text size
          legend.position = if (show_legend) "right" else "none") +  # Conditionally show legend
    facet_wrap(~model_type, scales = "free_y", nrow = 1)
  return(p)
}

# Generate individual plots with conditional width for the final plot
plots_baseline <- lapply(1:length(model_types_baseline), function(i) {
  data_subset <- data_baseline %>% filter(model_type == model_types_baseline[i])
  create_baseline_plot(data_subset, 
                       show_y_axis = (i == 1), 
                       show_legend = (i == length(model_types_baseline)))  # Show legend only on the last plot
})

# Combine the plots horizontally with adjusted widths
plot_widths_baseline <- rep(1, length(plots_baseline))  # Default width for all plots
plot_widths_baseline[length(plots_baseline)] <- 1.5  # Make the last plot wider to account for the legend

combined_plot_baseline <- plot_grid(plotlist = plots_baseline, nrow = 1, align = "h", rel_widths = plot_widths_baseline)

# Display the combined plot for baseline models
print(combined_plot_baseline)


