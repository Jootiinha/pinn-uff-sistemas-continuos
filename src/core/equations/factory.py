from typing import Dict, Any

from .base import BaseEquation
from .quadratic import QuadraticEquation, QuadraticParams
from .ode2_linear import ODE2LinearEquation, ODE2LinearParams
from .ode4_elasticity import ODE4thOrderEquation


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
