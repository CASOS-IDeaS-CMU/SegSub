# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)

# Load your dataset (ensure this path is correct for your data file)
data <- read.csv("data/1_vs_2_image_results_formatted.csv")

# Reshape the data similar to the approach in your provided code
data_long <- data %>%
  pivot_longer(cols = starts_with("img_"),
               names_to = "label_combination",
               values_to = "accuracy") %>%
  mutate(label_combination = gsub("img_|label_", "", label_combination)) %>%
  separate(label_combination, into = c("image_type", "label_type"), sep = "\\.") %>%
  mutate(
    image_type = factor(image_type, levels = c("old", "new"), labels = c("Original", "Modified")),
    label_type = factor(label_type, levels = c("old", "new")),
    model_name = case_when(
      grepl("Llava1.5", model_path) ~ "Llava1.5",
      grepl("Phi3", model_path) ~ "Phi3",
      grepl("Qwen2", model_path) ~ "Qwen2",
      TRUE ~ "Other"
    ),
    finetuned = ifelse(finetuned == 1, "Ft", "Base"),
    num_images = factor(num_images, levels = c(1, 2), labels = c("1 image", "2 images")),
    model_finetune_combination = factor(paste(model_name, finetuned, sep = "-"),
                                        levels = c("Llava1.5-Base", "Phi3-Base", "Qwen2-Base", 
                                                   "Llava1.5-Ft", "Phi3-Ft", "Qwen2-Ft"))
  )

# Define color scheme
model_colors <- c("Llava1.5-Base" = "#AFCFEA", "Phi3-Base" = "#C3E2B7", "Qwen2-Base" = "#F8D6A3",
                  "Llava1.5-Ft" = "#5187B2", "Phi3-Ft" = "#7FA560", "Qwen2-Ft" = "#E76D3C")

# Define color scheme
model_colors <- c("Llava1.5-Base" = "#AFCFEA", "Phi3-Base" = "#C3E2B7", "Qwen2-Base" = "#F8D6A3",
                  "Llava1.5-Ft" = "#5187B2", "Phi3-Ft" = "#7FA560", "Qwen2-Ft" = "#E76D3C")

# Function to create a plot with two groups (Original, Modified) each with 6 bars for model and finetuning status
create_plot <- function(data, show_y_axis = TRUE, show_legend = FALSE) {
  plot <- ggplot(data, aes(x = image_type, y = accuracy, fill = model_finetune_combination)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") +
    scale_fill_manual(values = model_colors, guide = guide_legend(nrow = 2, byrow = TRUE)) +  # Set legend to horizontal
    labs(x = NULL, y = if (show_y_axis) "Accuracy Score" else NULL, fill = if (show_legend) "Model: " else NULL) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal(base_size = 30) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 2, size = 22),
          axis.title.x = element_blank(),
          axis.title.y = if (show_y_axis) element_text(size = 26) else element_blank(),
          axis.text.y = if (show_y_axis) element_text(size = 26) else element_blank(),
          axis.ticks.y = if (show_y_axis) element_line(color = "black") else element_blank(),
          panel.grid.major = element_line(color = "gray", size = 0.5),
          panel.grid.minor = element_line(color = "gray", size = 0.25),
          strip.text = element_text(size = 28),
          panel.spacing.x = unit(1, "lines"),
          legend.title = element_text(size = 23),  
          legend.text = element_text(size = 23),
          legend.margin = if (show_legend) margin(t = 0, r = 0, b = 10, l = 0) else "none",
          legend.position = if (show_legend) "bottom" else "none") +  # Set legend position for horizontal layout
    facet_wrap(~num_images, scales = "fixed",  # Fixed width for all facets
               labeller = as_labeller(c(
                 "1 image" = "Single Image Questions",
                 "2 images" = "Two Image Questions"
               )))

  
  if (show_legend) {
    plot <- plot + theme(legend.background = element_rect(fill = "white", color = NA, size = 0.5, linetype = "solid"))
  }
  
  return(plot)
}



# Create a single plot to extract the legend
legend_plot <- create_plot(data_long, show_y_axis = TRUE, show_legend = TRUE)
legend <- get_legend(legend_plot)

# Generate the combined plot with both "1 image" and "2 images" facets, without a legend
main_plot <- create_plot(data_long, show_y_axis = TRUE, show_legend = FALSE)

# Add the extracted legend below the main plot
final_plot <- plot_grid(main_plot, legend, ncol = 1, rel_heights = c(1, 0.1))

# Display the final combined plot with the legend at the bottom
print(final_plot)
