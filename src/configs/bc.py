from dataclasses import dataclass
from typing import Callable

import torch


# Representação de BCs simples:
# - Dirichlet: y(x_b) = y_b
# - Neumann:   y'(x_b) = g_b
@dataclass
class DirichletBC:
    x_b: float
    y_b: float

@dataclass
class NeumannBC:
    x_b: float
    g_b: float  # valor de y'(x_b)

@dataclass
class StressBC:
    x_b: float
    stress_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    target: float

@dataclass
class MomentBC:
    moment_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    target: float
