library(ggplot2)
library(dplyr)
library(viridis)
library(tidyr)
library(scales)  # For wrap_format

data <- read.csv("data/context_scores.csv")

# Split and keep only the first segment of the model name
data_summary <- data %>%
  filter(type == "counterfactual",  # Filter for counterfactual type
         !is.na(context_score_perturbed) & !is.infinite(context_score_perturbed)) %>%
  group_by(dataset, model, context_score_perturbed) %>%
  summarize(mean_img_new_label_ret = mean(img_new_label_ret, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(model_first_segment = sapply(strsplit(model, "_"), `[`, 1))  # Split by underscore and take the first part

# Plot with different line types for each dataset and model combination
ggplot(data_summary, aes(x = context_score_perturbed, y = mean_img_new_label_ret,
                         group = interaction(dataset, model_first_segment), 
                         color = model_first_segment, linetype = dataset)) +
  geom_line(size = 1.5, alpha = 0.115) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k = 5), se = FALSE, size = 1.5, span = 0.1) +
  scale_color_discrete(labels = wrap_format(15)) +  # Wrap legend labels at 15 characters
  labs(title = "Counterfactual Accuracy vs Contextualization",
       x = "Context Cues",
       y = "Accuracy Score",
       color = "Model",
       linetype = "Dataset") +
  guides(
    color = guide_legend(order = 2),  # Set color legend order to appear first
    linetype = guide_legend(order = 1)  # Set linetype legend order to appear second
  ) +
  theme_minimal(base_size = 32) +
  theme(
    legend.background = element_rect(fill = rgb(1, 1, 1, alpha = 0.6), color = NA, size = 0.5, linetype = "solid"),
    legend.position = c(0.02, 0.065),                 # Position legend in bottom left corner
    legend.justification = c("left", "bottom"),# Anchor legend to bottom left
    legend.direction = "vertical",           # Arrange legend items horizontally
    legend.box = "horizontal",                   # Stack legends vertically
    legend.key.size = unit(1.5, "cm"),
    legend.title = element_text(size = 26),
    legend.text = element_text(size = 26),
    legend.margin = margin(0, 0, 0, 0),        # Remove padding around the legend
    legend.spacing = unit(0, "cm"),            # Remove spacing between legend items
    legend.box.spacing = unit(0, "cm"),        # Remove spacing between legend boxes
    plot.title = element_text(size = 32, hjust = 0.5),
    plot.margin = margin(0, 0, 0, 0)           # Remove padding around the plot borders
  )
