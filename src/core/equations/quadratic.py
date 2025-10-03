import math
from dataclasses import dataclass
from typing import Tuple

import torch

from .base import BaseEquation


@dataclass
class QuadraticParams:
    a: float
    b: float
    c: float


class QuadraticEquation(BaseEquation):
    """a*y^2 + b*y + c = 0 (tratando o desconhecido como y)."""
    def __init__(self, params: QuadraticParams):
        self.a = float(params.a)
        self.b = float(params.b)
        self.c = float(params.c)

    def residual_algebraic(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.a * y**2 + self.b * y + self.c

    def analytic_roots(self) -> Tuple[complex, complex]:
        a, b, c = self.a, self.b, self.c
        if abs(a) < 1e-15:
            if abs(b) < 1e-15:  # degenerado
                return (complex("nan"), complex("nan"))
            r = -c / b
            return (complex(r), complex("nan"))
        disc = b*b - 4*a*c
        if disc >= 0:
            r1 = (-b + math.sqrt(disc)) / (2*a)
            r2 = (-b - math.sqrt(disc)) / (2*a)
            return (complex(r1), complex(r2))
        else:
            real = -b / (2*a)
            imag = math.sqrt(-disc) / (2*a)
            return (complex(real, imag), complex(real, -imag))
