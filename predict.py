import torch
from safetensors.torch import load_file
from pinn import PINN

model = PINN(n_hidden=20)
model.load_state_dict(load_file("model/model.safetensors"))
model.eval()

with torch.no_grad():
    print(model(torch.tensor([[1.5]])).item())