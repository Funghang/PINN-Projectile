import copy
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors.torch import save_file

from pinn import (PINN, true_solution, data_loss,
                  physics_loss, initial_condition_loss)

# ---------------- Config ----------------
g, h0, v0 = 9.8, 1.0, 10.0
t_min, t_max = 0.0, 2.0
N_data = 1000
noise_level = 0.7

lambda_data, lambda_ode, lambda_ic = 1.0, 9.0, 1.0
lr = 0.01
num_epochs = 5000
print_every = 200
patience_steps = 600
save_dir = "model"

# ---------------- Data ----------------
np.random.seed(0)
torch.manual_seed(0)

t_data = np.linspace(t_min, t_max, N_data)
h_noisy = true_solution(t_data, g, h0, v0) + noise_level * np.random.randn(N_data)

t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
h_tensor = torch.tensor(h_noisy, dtype=torch.float32).view(-1, 1)

# ---------------- Model ----------------
model = PINN(n_hidden=20)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------- Training ----------------
best_loss, best_state, best_step, no_decrease = float("inf"), None, 0, 0

model.train()
for step in range(1, num_epochs + 1):
    optimizer.zero_grad()
    l_data = data_loss(model, t_tensor, h_tensor)
    l_ode = physics_loss(model, t_tensor, g, v0)
    l_ic = initial_condition_loss(model, h0)
    loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic
    loss.backward()
    optimizer.step()

    current = loss.item()
    if current < best_loss:
        best_loss, best_step = current, step
        best_state = copy.deepcopy(model.state_dict())
        no_decrease = 0
    else:
        no_decrease += 1

    if step % print_every == 0:
        print(f"Step {step}/{num_epochs}, Total={current:.6f}, "
              f"Data={l_data.item():.6f}, ODE={l_ode.item():.6f}, "
              f"IC={l_ic.item():.6f}")

    if no_decrease >= patience_steps:
        print(f"Early stopping at step {step}. Best loss {best_loss:.6f} at step {best_step}.")
        break

model.load_state_dict(best_state)
model.eval()

# Save 
os.makedirs(save_dir, exist_ok=True)
save_file(model.state_dict(), os.path.join(save_dir, "model.safetensors"))
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump({
        "n_hidden": 20,
        "physics": {"g": g, "h0": h0, "v0": v0},
        "loss_weights": {"lambda_data": lambda_data,
                         "lambda_ode": lambda_ode,
                         "lambda_ic": lambda_ic},
        "best_step": best_step,
        "best_total_loss": best_loss,
    }, f, indent=2)
print(f"Saved model to '{save_dir}/'")

# Plot 
t_plot = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
with torch.no_grad():
    h_pred = model(torch.tensor(t_plot)).numpy()

plt.figure(figsize=(8, 5))
plt.scatter(t_data, h_noisy, color="red", label="Noisy Data")
plt.plot(t_plot, true_solution(t_plot, g, h0, v0), "b--", label="Exact Solution")
plt.plot(t_plot, h_pred, "y--", label="PINN Prediction")
plt.xlabel("Time (t)")
plt.ylabel("Height h(t)")
plt.legend()
plt.grid(True)
plt.title("Height Prediction with PINN")
plt.savefig("results.png", dpi=150)
plt.show()