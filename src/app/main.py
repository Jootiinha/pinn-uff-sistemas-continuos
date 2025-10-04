import math
import random
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.configs.bc import DirichletBC, NeumannBC,StressBC
from src.configs.train_configs import TrainConfigAlgebraic, TrainConfigODE2
from src.core.equations import EquationFactory, QuadraticParams, ODE2LinearParams,  PDEEq
from src.core.solvers import PINNAlgebraicSolver, PINNODE2Solver, PINNODE4Solver
from src.core import graphs

def main():
    a = 2.0     # a=r
    b = 1.0     #b=r

    device = "cpu"
 
    #TODO preciso desse param ?
    # r = lambda x: x          
    # ode_params = PDEParams(x=r)
    # eq_ode = EquationFactory.create("pde_equation", params=ode_params)
    eq_ode = EquationFactory.create("pde_equation", params=None)

    bcs = [
        StressBC(x_b=a, stress_fn=PDEEq.trr, target=0.0),       #condição de contorno Trr = 0
        StressBC(x_b=b, stress_fn=PDEEq.ttt, target=0.0),       #condição de contorno Ttt = 0
    ]
    
    ode_cfg = TrainConfigODE2(
        epochs=6000,
        n_collocation=512,
        lr=1e-3,
        hidden=64,
        depth=4,
        device=device,
        domain=(a, b),
        w_pde=1.0,
        w_bc=1.0,
        normalize_x=False
    )
    ode_solver = PINNODE4Solver(eq_ode, ode_cfg, bcs)

    ode_solver.model._init()
    
    print("\nTreinando PINN (EDO 4ª ordem) para var4phi = 0, Trr(a)=0, Ttt(b)=0 ...")
    ode_solver.train(verbose_every=500)
    graphs.create_trainning_graph()

    # Avaliar rede no domínio
    rs = torch.linspace(a, b, 200).view(-1,1)
    phi_pred = ode_solver.predict(rs)
    graphs.create_radius_graph(rs, phi_pred)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
