from src.configs.bc import DirichletBC, StressBC
from src.configs.train_configs import TrainConfigODE4, TrainConfigODE2
from src.core.equations.factory import EquationFactory

# Domínio
a = 68.073
b = 73.328

# Dispositivo de treino
device = "cpu"

# Equação
equation = EquationFactory.create("ode4_elasticity")

# Configuração do Treino
train_config = TrainConfigODE4(
    epochs=2000,
    n_collocation=256,
    lr=1e-3,
    hidden=64,
    depth=4,
    device=device,
    domain=(a, b),
    w_pde=1.0,
    w_bc=1.0,
    normalize_x=True,
    y_a=68.073,
    y_b=73.328,
)


# Condições de Contorno
bcs = [
    StressBC(x_b=a, stress_fn=equation.trr, target=0.0),
    StressBC(x_b=b, stress_fn=equation.ttt, target=0.0),
]
