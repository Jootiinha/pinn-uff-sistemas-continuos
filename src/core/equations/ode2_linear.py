from dataclasses import dataclass
from typing import Callable

import torch

from .base import BaseEquation


@dataclass
class ODE2LinearParams:
    p: Callable[[torch.Tensor], torch.Tensor]  # p(x)
    q: Callable[[torch.Tensor], torch.Tensor]  # q(x)
    r: Callable[[torch.Tensor], torch.Tensor]  # r(x)


class ODE2LinearEquation(BaseEquation):
    def __init__(self, params: ODE2LinearParams):
        self.p_fun = params.p
        self.q_fun = params.q
        self.r_fun = params.r

    def residual_ode2(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        # todos como tensores (N, 1)
        p = self.p_fun(x)
        q = self.q_fun(x)
        r = self.r_fun(x)
        return d2y + p * dy + q * y - r
