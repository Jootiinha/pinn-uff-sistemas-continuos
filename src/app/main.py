import math
import random

import torch

from src.configs.bc import DirichletBC, NeumannBC
from src.configs.train_configs import TrainConfigAlgebraic, TrainConfigODE2
from src.core.equations import EquationFactory, QuadraticParams, ODE2LinearParams
from src.core.solvers import PINNAlgebraicSolver, PINNODE2Solver


def main():
    # -----------------------
    # (A) Algébrico: x^2 - 5x + 6 = 0  -> raízes 2 e 3
    # -----------------------
    quad_params = QuadraticParams(a=1.0, b=-5.0, c=6.0)
    eq_quad = EquationFactory.create("quadratic", params=quad_params)
    alg_cfg = TrainConfigAlgebraic(epochs=2000, device="cpu")
    alg_solver = PINNAlgebraicSolver(eq_quad, alg_cfg)
    print("Treinando PINN (algébrico) para raízes de 2º grau...")
    alg_solver.train(verbose_every=400)
    print("Raízes analíticas:", eq_quad.analytic_roots())

    # -----------------------
    # (B) EDO 2ª ordem: y'' + y = 0
    # Condições: y(0)=0 (Dirichlet) e y'(0)=1 (Neumann) -> y = sin(x)
    # Domínio: [0, pi/2]
    # -----------------------
    device = "cpu"
    p = lambda x: torch.zeros_like(x)         # p(x)=0
    q = lambda x: torch.ones_like(x)          # q(x)=1
    r = lambda x: torch.zeros_like(x)         # r(x)=0
    ode_params = ODE2LinearParams(p=p, q=q, r=r)
    eq_ode = EquationFactory.create("ode2_linear", params=ode_params)

    bcs = [
        DirichletBC(x_b=0.0, y_b=0.0),  # y(0)=0
        NeumannBC(x_b=0.0, g_b=1.0),    # y'(0)=1
    ]
    ode_cfg = TrainConfigODE2(
        epochs=6000,
        n_collocation=256,
        lr=1e-3,
        hidden=64,
        depth=4,
        device=device,
        domain=(0.0, math.pi/2),
        w_pde=1.0,
        w_bc=1.0,
        normalize_x=True
    )
    ode_solver = PINNODE2Solver(eq_ode, ode_cfg, bcs)
    print("\nTreinando PINN (EDO 2ª ordem) para y'' + y = 0, y(0)=0, y'(0)=1 ...")
    ode_solver.train(verbose_every=500)

    # Avaliação em uma malha
    xs = torch.linspace(0.0, math.pi/2, 101)
    y_pred = ode_solver.predict(xs)
    # Comparação com sin(x)
    y_true = torch.sin(xs).cpu()
    max_abs_err = (y_pred - y_true).abs().max().item()
    print(f"Erro máximo |y_pred - sin(x)| no domínio: {max_abs_err:.3e}")

    # Dica: plote xs vs y_pred para inspecionar a solução.


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
