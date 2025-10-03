import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable

import torch


class BaseEquation:
    """Interface/base para equações.
    O solver pode chamar .residual(...) com assinaturas diferentes.
    Para manter compatibilidade, aceitamos **kwargs.
    """
    def solution_dim(self) -> int:
        return 1

    # Para problemas ALGÉBRICOS (ex.: encontrar raízes), usamos y (constante)
    def residual_algebraic(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Para EDO de 2ª ordem, usamos x, y(x), y'(x), y''(x)
    def residual_ode2(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ------------------------------
# 1.a) Equação algébrica: 2º grau
# ------------------------------

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


# ------------------------------
# 1.b) EDO 2ª ordem linear geral
# y'' + p(x) y' + q(x) y = r(x)
# ------------------------------

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


# ------------------------------
# 1.c) EDO de 4ª ordem para elasticidade
# ------------------------------

class ODE4thOrderEquation(BaseEquation):
    """
    Define a equação diferencial de 4ª ordem e as grandezas físicas
    (tensões) associadas.
    """
    def residual_ode4(self, x: torch.Tensor, d2y: torch.Tensor, d3y: torch.Tensor, d4y: torch.Tensor) -> torch.Tensor:
        """
        Resíduo da EDO: d^4(phi)/dr^4 = 0
        A formulação é complexa e baseada em uma função auxiliar 'g'.
        """
        # Esta é a formulação que estava no solver original.
        # g = y'' + y' (não usado diretamente aqui, mas é a base)
        # O resíduo é d/dx(g/x) * d2g/dx2
        # Isso pode ser reescrito em termos de derivadas de y.
        # d/dx( (y''+y')/x ) * d2/dx2(y''+y')
        # (x(y'''+y'') - (y''+y'))/x^2 * (y''''+y''')
        
        # A implementação original no solver era:
        # g = d2y+dy
        # dg_dx_x= self._grad(g/x,x)
        # d2g_dx = self._grad(self._grad(g,x),x)
        # resid = dg_dx_x * d2g_dx
        #
        # Re-expressando em termos de derivadas de y:
        # dg_dx_x = (x * (d3y + d2y) - (d2y + dy)) / x**2
        # d2g_dx = d4y + d3y
        # resid = dg_dx_x * d2g_dx
        #
        # Para simplificar e evitar o dy, vamos usar a formulação d4y=0
        return d4y

    def trr(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """ Tensão radial: Trr = (1/r) * d(phi)/dr """
        return dy / x

    def ttt(self, x: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        """ Tensão tangencial: Ttt = d2(phi)/dr2 """
        return d2y

# ------------------------------
# Factory
# ------------------------------

class EquationFactory:
    _REGISTRY: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, ctor):
        cls._REGISTRY[name] = ctor

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEquation:
        if name not in cls._REGISTRY:
            raise ValueError(f"Equação '{name}' não registrada.")
        return cls._REGISTRY[name](**kwargs)


EquationFactory.register("quadratic", lambda params: QuadraticEquation(params=params))
EquationFactory.register("ode2_linear", lambda params: ODE2LinearEquation(params=params))
EquationFactory.register("ode4_elasticity", lambda **kwargs: ODE4thOrderEquation())
