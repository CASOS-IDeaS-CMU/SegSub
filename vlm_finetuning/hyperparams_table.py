import json

# JSON data (truncated here for brevity)
data = json.load(open("configs.json", "r"))

# Updated list of important hyperparameters
parameters = [
    "hidden_size",
    "hidden_act", 
    "intermediate_size", 
    "num_attention_heads", 
    "num_hidden_layers", 
    "vision_model",
    "image_embedding_dim",
    "vocab_size", 
    "max_position_embeddings", 
    "torch_dtype",
    "attention_dropout",
    "initializer_range",
    "sliding_window",
    "temperature",
]

# Function to get parameter value
def get_param(config, param):
    return config.get(param, "N/A")

# Create LaTeX table
latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|l|c|c|c|}\n\\hline\n"
latex_table += "Hyperparameter & Phi3V & Qwen2VL & Llava \\\\\n\\hline\n"

for param in parameters:
    phi3v_value = get_param(data["Phi3VConfig"], param)
    qwen2vl_value = get_param(data["Qwen2VLConfig"], param)
    llava_value = get_param(data["LlavaConfig"], param)
    
    latex_table += f"{param} & {phi3v_value} & {qwen2vl_value} & {llava_value} \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\caption{Important hyperparameters for the models}\n\\end{table}"

print(latex_table)
