import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, n_hidden=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, t):
        return self.net(t)


def true_solution(t, g=9.8, h0=1.0, v0=10.0):
    """Exact height h(t) = h0 + v0*t - 0.5*g*t^2."""
    return h0 + v0 * t - 0.5 * g * t**2


def derivative(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def data_loss(model, t_data, h_data):
    return torch.mean((model(t_data) - h_data) ** 2)


def physics_loss(model, t, g=9.8, v0=10.0):
    t = t.detach().clone().requires_grad_(True)
    h_pred = model(t)
    dh_dt_pred = derivative(h_pred, t)
    dh_dt_true = v0 - g * t
    return torch.mean((dh_dt_pred - dh_dt_true) ** 2)


def initial_condition_loss(model, h0=1.0):
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    return (model(t0) - h0).pow(2).mean()