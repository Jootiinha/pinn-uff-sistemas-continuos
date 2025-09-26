import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfigAlgebraic:
    epochs: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    branches: int = 2
    hidden: int = 64
    depth: int = 3
    device: str = "cpu"
    y_clip: float = 1e6
    w_residual: float = 1.0
    w_diversity: float = 0.02
    w_range: float = 0.0
    y_minmax: Tuple[float, float] = (-1e3, 1e3)


@dataclass
class TrainConfigODE2:
    epochs: int = 6000
    n_collocation: int = 256
    lr: float = 1e-3
    hidden: int = 64
    depth: int = 4
    device: str = "cpu"
    domain: Tuple[float, float] = (0.0, math.pi/2)  # [x0, x1]
    w_pde: float = 1.0
    w_bc: float = 1.0
    # opcional: normalização do domínio para estabilidade
    normalize_x: bool = True
