# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)

# Define the dataset
data <- read.csv("data/negative_eval.csv")


# Reshape the data into long format
data_long <- data %>%
  pivot_longer(cols = c(webqa, vqa, okvqa), names_to = "metric", values_to = "accuracy") %>%
  mutate(
    model_name = case_when(
      grepl("qwen2", model_path) ~ "Qwen2",
      grepl("phi3", model_path) ~ "Phi3",
      grepl("llava15", model_path) ~ "Llava1.5",
      TRUE ~ "Other"
    ),
    finetuned = ifelse(grepl("-ft", model_path), "Ft", "Base"),
    model_finetune_combination = factor(paste(model_name, finetuned, sep = "-"),
                                        levels = c("Llava1.5-Base", "Phi3-Base", "Qwen2-Base", 
                                                   "Llava1.5-Ft", "Phi3-Ft", "Qwen2-Ft"))
  )

# Define color scheme
model_colors <- c("Llava1.5-Base" = "#AFCFEA", "Phi3-Base" = "#C3E2B7", "Qwen2-Base" = "#F8D6A3",
                  "Llava1.5-Ft" = "#5187B2", "Phi3-Ft" = "#7FA560", "Qwen2-Ft" = "#E76D3C")

# Create a function to generate individual plots for each metric
create_metric_plot <- function(metric_name, data) {
  ggplot(data %>% filter(metric == metric_name), aes(x = model_finetune_combination, y = accuracy, fill = model_finetune_combination)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") +
    scale_fill_manual(values = model_colors) +
    labs(title = metric_name, x = NULL, y = "Accuracy Score") +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal(base_size = 18) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
          axis.title.x = element_blank(),
          axis.title.y = element_text(size = 18),
          axis.text.y = element_text(size = 14),
          panel.grid.major = element_line(color = "gray", size = 0.5),
          panel.grid.minor = element_line(color = "gray", size = 0.25),
          plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
          legend.position = "none")
}

# Generate individual plots
webqa_plot <- create_metric_plot("webqa", data_long)
vqa_plot <- create_metric_plot("vqa", data_long)
okvqa_plot <- create_metric_plot("okvqa", data_long)

# Combine the plots into one row
final_plot <- plot_grid(webqa_plot, vqa_plot, okvqa_plot, ncol = 3, align = "h", labels = c("A", "B", "C"))

# Display the final plot
print(final_plot)
