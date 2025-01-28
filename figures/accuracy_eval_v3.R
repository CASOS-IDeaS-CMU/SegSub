# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)  # For combining individual plots

# Load your dataset (ensure this path is correct for your data file)
data <- read.csv("data/all_results_formatted_2.csv")

# Filter out rows with the model type 'GPT-4o-mini'
data_filtered <- data %>%
  filter(model_path != "GPT-4o-mini")

data_long <- data_filtered %>%
  pivot_longer(cols = starts_with("img_"),
               names_to = "label_type",
               values_to = "accuracy") %>%
  mutate(label_type = gsub("img_|label_", "", label_type)) %>%
  separate(label_type, into = c("image_type", "label_type"), sep = "\\.") %>%
  mutate(
    image_type = factor(image_type, levels = c("old", "new")),
    label_type = factor(label_type, levels = c("old", "new")),
    model = ifelse(finetuned == 1, "Ft", "Base"),
    model_type = model_path  # Ensure model_type is available for filtering
  )

# Now filter for the specific combinations you care about
data_long <- data_long %>%
  filter(
    (model == "Base" & image_type == "old" & label_type == "old") |
      (model == "Base" & image_type == "new" & label_type == "new") |
      (model == "Ft" & image_type == "old" & label_type == "old") |
      (model == "Ft" & image_type == "new" & label_type == "new")
  )

# Create a new column for grouping by combination of dataset and type
data_long <- data_long %>%
  mutate(dataset_type = factor(interaction(dataset, type), levels = c(
    "VQA.counter", "OKVQA.counter", "WebQA.counter", "WebQA.conf"
  )))  # Set custom order for x-axis

# Combine model and model_type into one factor for unique colors per bar, with specified ordering
data_long$model_label_combination <- factor(
  paste0(data_long$model_type, "-", data_long$model),
  levels = c("Llava1.5-Base", "Phi3-Base", "Qwen2-Base", "Llava1.5-Ft", "Phi3-Ft", "Qwen2-Ft")
)

model_colors <- c("Llava1.5-Base" = "#AFCFEA", "Phi3-Base" = "#C3E2B7", "Qwen2-Base" = "#F8D6A3",
                  "Llava1.5-Ft" = "#5187B2", "Phi3-Ft" = "#7FA560", "Qwen2-Ft" = "#E76D3C")


# Define the color palette for each unique model/model_type combination
# model_colors <- c("Llava1.5-Base" = "#A3C1DA", "Phi3-Base" = "#B5D3B6", "Qwen2-Base" = "#F1A9A0",
#                   "Llava1.5-Ft" = "#467098", "Phi3-Ft" = "#6E9B6C", "Qwen2-Ft" = "#D9534F")

# model_colors <- c("Llava1.5-Base" = "#D0E6F2", "Phi3-Base" = "#E4F1E2", "Qwen2-Base" = "#FFE5D5",
#                   "Llava1.5-Ft" = "#77AADD", "Phi3-Ft" = "#88CCAA", "Qwen2-Ft" = "#FFAA77")

# model_colors <- c("Llava1.5-Base" = "#CAE7E7", "Phi3-Base" = "#CDE0B5", "Qwen2-Base" = "#FFD1BA",
#                   "Llava1.5-Ft" = "#4DA6A6", "Phi3-Ft" = "#77BB44", "Qwen2-Ft" = "#FF6347")


# Define specific names for each dataset_type and label_type combination
data_long <- data_long %>%
  mutate(label_type_custom = case_when(
    dataset_type == "VQA.counter" & label_type == "old" ~ "Original",
    dataset_type == "VQA.counter" & label_type == "new" ~ "Counter",
    dataset_type == "OKVQA.counter" & label_type == "old" ~ "Original",
    dataset_type == "OKVQA.counter" & label_type == "new" ~ "Counter",
    dataset_type == "WebQA.counter" & label_type == "old" ~ "Original",
    dataset_type == "WebQA.counter" & label_type == "new" ~ "Counter",
    dataset_type == "WebQA.conf" & label_type == "old" ~ "Original",
    dataset_type == "WebQA.conf" & label_type == "new" ~ "Conflicts",
    TRUE ~ as.character(label_type)  # Default case (shouldn't be used if all cases are specified)
  ))

data_long <- data_long %>%
  mutate(label_type_custom = factor(
    label_type_custom,
    levels = c("Original","Counter","Conflicts")
  ))


# Updated create_plot function to use label_type_custom
create_plot <- function(data, show_y_axis = TRUE, show_legend = FALSE) {
  p <- ggplot(data, aes(x = label_type_custom, y = accuracy, fill = model_label_combination)) +
    geom_bar(stat = "identity", position = "dodge", color = "black") +
    scale_fill_manual(values = model_colors) +
    labs(x = NULL, y = if (show_y_axis) "Accuracy Score" else NULL, fill = if (show_legend) "Model - Type") +
    scale_y_continuous(limits = c(0, 1)) +
    guides(fill = guide_legend(nrow = 1)) +  # Arrange legend items in a single row
    theme_minimal(base_size = 20) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5, vjust=1, size=17),
          axis.title.x = element_blank(),
          axis.title.y = if (show_y_axis) element_text(size = 20) else element_blank(),
          axis.text.y = if (show_y_axis) element_text(size = 20) else element_blank(),
          axis.ticks.y = if (show_y_axis) element_line(color = "black") else element_blank(),
          panel.grid.major = element_line(color = "gray", size = 0.5),
          panel.grid.minor = element_line(color = "gray", size = 0.25),
          strip.text = element_text(size = 20),
          panel.spacing.x = unit(1, "lines"),
          legend.title = element_text(size = 17),  
          legend.text = element_text(size = 17),   
          legend.position = if (show_legend) "bottom" else "none",  # Position legend at the bottom
          legend.background = element_rect(fill = rgb(1, 1, 1, alpha = 0.3), color = NA, size = 0.5, linetype = "solid"),
          legend.margin = margin(0, 0, 0, 0),
          legend.spacing = unit(0, "cm"),
          legend.box.spacing = unit(0, "cm")) + 
    facet_wrap(~dataset_type, scales = "free_y", nrow = 1,
               labeller = as_labeller(c(
                 "VQA.counter" = "VQA",
                 "OKVQA.counter" = "OKVQA",
                 "WebQA.counter" = "WebQA",
                 "WebQA.conf" = "WebQA"
               )))  # Custom labels for dataset_type facets
  return(p)
}


# Create a single plot to extract the legend
dataset_types <- levels(data_long$dataset_type)
legend_plot <- create_plot(data_long %>% filter(dataset_type == dataset_types[1]), show_y_axis = TRUE, show_legend = TRUE)
legend <- get_legend(legend_plot)

# Generate individual plots without legends
plots <- lapply(1:length(dataset_types), function(i) {
  data_subset <- data_long %>% filter(dataset_type == dataset_types[i])
  create_plot(data_subset, 
              show_y_axis = (i == 1), 
              show_legend = FALSE)  # No legend in individual plots
})

# Combine the main plots horizontally with adjusted widths
plot_widths <- rep(1, length(plots))  # Default width for all plots
plot_widths[1] <- 1.3
combined_plots <- plot_grid(plotlist = plots, nrow = 1, align = "h", rel_widths = plot_widths)

# Add the extracted legend below the combined plot
final_plot <- plot_grid(combined_plots, legend, ncol = 1, rel_heights = c(1, 0.1))

# Display the final combined plot with the legend at the bottom
print(final_plot)
