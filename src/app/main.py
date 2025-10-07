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
    a = 1.0    #  a = r
    b = 60.0    #  b = r

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
        epochs=4500,
        n_collocation=512,
        lr=1e-3,
        hidden=64,
        depth=4,
        device=device,
        domain=(a, b),
        w_pde=1.0,
        w_bc=1.8,
        normalize_x=False
    )
    ode_solver = PINNODE4Solver(eq_ode, ode_cfg, bcs)

    ode_solver.model._init()
    
    print("\nTreinando PINN (EDO 4ª ordem) para var4phi = 0, Trr(a)=0, Ttt(b)=0 ...")
    ode_solver.train(verbose_every=500)
    graphs.create_trainning_graph()

    # Avaliar rede no domínio
    rs = torch.linspace(a, b, 200).view(-1,1)

    phi_pred, trr_pred, ttt_pred,momento_val = ode_solver.predict_with_stress(rs)

    trr_pred_M = trr_pred * 4*momento_val
    trr_analitico = PDEEq.T_rr_analytical(rs,a,b,momento_val)

    graphs.create_radius_graph(rs, phi_pred)
    graphs.create_stress_graph(rs, phi_pred, trr_pred, ttt_pred)
    # graphs.create_moment_graph(rs, momento_val)
    graphs.create_trr_analitico_vs_predito_graph(rs,trr_analitico,trr_pred_M)
    graphs.create_trr_catia()

    graphs.create_pinn_vs_analytic_report(
        rs_pinn=rs, 
        trr_analitico=trr_analitico, 
        trr_pred=trr_pred_M
    )

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
