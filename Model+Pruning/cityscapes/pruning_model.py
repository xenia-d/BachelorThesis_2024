import torch
from netslim import prune, load_pruned_model
from models import SegNet, SegNetMtan 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_model(model_class, weights_path):
    model = model_class()
    loaded_model = torch.load(weights_path, map_location=torch.device('cpu'))
    if isinstance(loaded_model, dict):
        return load_pruned_model(model, loaded_model)
    return loaded_model

model = SegNetMtan()

# Count parameters
total_params_famo = count_parameters(model)
print("Total parameters in the model before pruning:", total_params_famo)

# Load the pruned weights into the model
pruned_model = load_model(SegNetMtan, 'experiments/cityscapes/model_compressed_weights.pth')

pruned_model = prune(pruned_model, (128, 256, 3)) 

# Count parameters after pruning
total_params_pruned = count_parameters(pruned_model)
print("Total parameters in the model after pruning:", total_params_pruned)