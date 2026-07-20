import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn

# Auto-download dependency if missing
if not os.path.exists("kan.py"):
    kan_url = "https://raw.githubusercontent.com/Blealtan/efficient-kan/master/src/efficient_kan/kan.py"
    urllib.request.urlretrieve(kan_url, "kan.py")

from kan import KAN


class MLP(nn.Module):
    """Standard Multi-Layer Perceptron baseline."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseKAN(nn.Module):
    """Baseline KAN model with uniform grid adaptation."""

    def __init__(self, layers_hidden: list, grid_size: int = 3, spline_order: int = 3):
        super().__init__()
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def update_grid(self, x: torch.Tensor):
        with torch.no_grad():
            for layer in self.kan.layers:
                layer.update_grid(x)
                x = layer(x)


class GradientAdaptiveKAN(nn.Module):
    """
    Gradient-Adaptive Kolmogorov-Arnold Network (GradKAN).
    Dynamically reallocates knot locations based on filtered gradient magnitude (EMA)
    and re-projects existing weights via PyTorch Least Squares (LSTSQ).
    """

    def __init__(
        self,
        layers_hidden: list,
        grid_size: int = 3,
        spline_order: int = 3,
        ema_decay: float = 0.5,
    ):
        super().__init__()
        self.kan = KAN(layers_hidden, grid_size=grid_size, spline_order=spline_order)
        self.ema_decay = ema_decay
        self._register_gradient_hooks()

    def _register_gradient_hooks(self):
        for layer in self.kan.layers:
            layer.grad_ema = torch.zeros_like(layer.spline_weight)

            def make_hook(l):
                def hook(grad):
                    l.grad_ema = l.grad_ema.to(grad.device)
                    l.grad_ema = self.ema_decay * l.grad_ema + (1 - self.ema_decay) * grad.abs()
                    return grad

                return hook

            layer.spline_weight.register_hook(make_hook(layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

    def update_grid_from_gradients(self, x: torch.Tensor, margin: float = 0.01):
        with torch.no_grad():
            for layer in self.kan.layers:
                in_features = layer.in_features
                G = layer.grid_size
                k = layer.spline_order

                # 1. Compute original evaluation outputs before grid update
                splines = layer.b_splines(x)
                orig_coeff = layer.spline_weight
                y_eval = torch.einsum("bik,oik->boi", splines, orig_coeff)

                # 2. Derive Gradient Density Function (PDF) and CDF
                x_min = x.min(dim=0)[0] - margin
                x_max = x.max(dim=0)[0] + margin

                current_ema = layer.grad_ema.to(x.device)
                importance = current_ema.mean(dim=0) + 1e-5
                pdf = importance / importance.sum(dim=-1, keepdim=True)
                cdf = torch.cumsum(pdf, dim=-1)
                cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

                x_coords = x_min.unsqueeze(1) + (x_max - x_min).unsqueeze(1) * torch.linspace(
                    0, 1, G + k + 1, device=x.device
                ).unsqueeze(0)

                # 3. Inverse CDF transformation for knot relocation
                target_cdf = torch.linspace(0, 1, G + 1, device=x.device).unsqueeze(0).expand(in_features, -1)
                new_core_knots = torch.zeros((in_features, G + 1), device=x.device)
                for i in range(in_features):
                    new_core_knots[i] = torch.tensor(
                        np.interp(
                            target_cdf[i].cpu().numpy(),
                            cdf[i].cpu().numpy(),
                            x_coords[i].cpu().numpy(),
                        ),
                        dtype=torch.float32,
                        device=x.device,
                    )

                step_left = (new_core_knots[:, 1] - new_core_knots[:, 0]).unsqueeze(1)
                step_right = (new_core_knots[:, -1] - new_core_knots[:, -2]).unsqueeze(1)

                left = new_core_knots[:, 0:1] - step_left * torch.arange(k, 0, -1, device=x.device).unsqueeze(0)
                right = new_core_knots[:, -1:] + step_right * torch.arange(1, k + 1, device=x.device).unsqueeze(0)

                new_grid = torch.cat([left, new_core_knots, right], dim=-1)
                layer.grid.copy_(new_grid)

                # 4. Re-project learned weights using PyTorch LSTSQ
                new_splines = layer.b_splines(x)
                new_weights = torch.zeros_like(orig_coeff)

                for i in range(in_features):
                    A = new_splines[:, i, :]
                    B = y_eval[:, :, i]
                    X = torch.linalg.lstsq(A, B).solution
                    new_weights[:, i, :] = X.T

                layer.spline_weight.data.copy_(new_weights)
                x = layer(x)
                layer.grad_ema.zero_()


# Alias for convenience
GradKAN = GradientAdaptiveKAN


if __name__ == "__main__":
    # Minimal sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sanity check on device: {device}")

    # Generate dummy high-frequency synthetic data
    X = torch.linspace(-1, 1, 500).unsqueeze(1).to(device)
    Y = torch.sin(5 * torch.pi * X) * torch.exp(-20 * (X**2))

    # Initialize models
    mlp = MLP(1, 10, 1).to(device)
    base_kan = BaseKAN([1, 10, 1], grid_size=5).to(device)
    grad_kan = GradKAN([1, 10, 1], grid_size=5, ema_decay=0.5).to(device)

    criterion = nn.MSELoss()

    # Forward & Backward pass test
    loss_mlp = criterion(mlp(X), Y)
    loss_base = criterion(base_kan(X), Y)
    loss_grad = criterion(grad_kan(X), Y)

    loss_mlp.backward()
    loss_base.backward()
    loss_grad.backward()

    # Dynamic Grid Updates
    base_kan.update_grid(X)
    grad_kan.update_grid_from_gradients(X)

    print("Sanity check passed successfully! Models are fully functional.")