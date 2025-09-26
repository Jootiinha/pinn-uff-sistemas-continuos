from dataclasses import dataclass


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
