import torch
from torch import nn


class MLPEncoder(nn.Module):
    """
    MLP encoder. Initialized to be essentially linear.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Initialize linear layers to be identity-like
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.5)
        self.fc2 = nn.Linear(input_dim, output_dim)
        nn.init.eye_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.nonlinearity(self.fc1(x))
        x = self.fc2(x)
        return x
