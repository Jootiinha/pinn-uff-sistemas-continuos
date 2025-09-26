import torch
import torch.nn as nn


class MLPBranches(nn.Module):
    """MLP para problema ALGÉBRICO: recebe t (dummy) e retorna K saídas (ramos)."""
    def __init__(self, in_dim=1, hidden=64, depth=3, branches=2, out_dim_per_branch=1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.Tanh()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(dim, branches * out_dim_per_branch)
        self.branches = branches
        self.out_dim_per_branch = out_dim_per_branch
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.net(t)
        y = self.head(h)
        return y.view(-1, self.branches, self.out_dim_per_branch)


class MLP1D(nn.Module):
    """MLP para EDO: recebe x e retorna y(x)."""
    def __init__(self, in_dim=1, hidden=64, depth=4, out_dim=1, act=nn.Tanh):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), act()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(dim, out_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))
