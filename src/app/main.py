import math
import random
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.configs.bc import DirichletBC, NeumannBC,StressBC
from src.configs.train_configs import TrainConfigAlgebraic, TrainConfigODE2
from src.core.equations import EquationFactory, QuadraticParams, ODE2LinearParams, PDEParams, PDEEq
from src.core.solvers import PINNAlgebraicSolver, PINNODE2Solver, PINNODE4Solver


def main():
    # # -----------------------
    # # (A) Algébrico: x^2 - 5x + 6 = 0  -> raízes 2 e 3
    # # -----------------------
    # quad_params = QuadraticParams(a=1.0, b=-5.0, c=6.0)
    # eq_quad = EquationFactory.create("quadratic", params=quad_params)
    # alg_cfg = TrainConfigAlgebraic(epochs=2000, device="cpu")
    # alg_solver = PINNAlgebraicSolver(eq_quad, alg_cfg)
    # print("Treinando PINN (algébrico) para raízes de 2º grau...")
    # alg_solver.train(verbose_every=400)
    # print("Raízes analíticas:", eq_quad.analytic_roots())

    # # -----------------------
    # # (B) EDO 2ª ordem: y'' + y = 0
    # # Condições: y(0)=0 (Dirichlet) e y'(0)=1 (Neumann) -> y = sin(x)
    # # Domínio: [0, pi/2]
    # # -----------------------
    # device = "cpu"
    # p = lambda x: torch.zeros_like(x)         # p(x)=0
    # q = lambda x: torch.ones_like(x)          # q(x)=1
    # r = lambda x: torch.zeros_like(x)         # r(x)=0
    # ode_params = ODE2LinearParams(p=p, q=q, r=r)
    # eq_ode = EquationFactory.create("ode2_linear", params=ode_params)

    # bcs = [
    #     DirichletBC(x_b=0.0, y_b=0.0),  # y(0)=0
    #     NeumannBC(x_b=0.0, g_b=1.0),    # y'(0)=1
    # ]
    # ode_cfg = TrainConfigODE2(
    #     epochs=6000,
    #     n_collocation=256,
    #     lr=1e-3,
    #     hidden=64,
    #     depth=4,
    #     device=device,
    #     domain=(0.0, math.pi/2),
    #     w_pde=1.0,
    #     w_bc=1.0,
    #     normalize_x=True
    # )
    # ode_solver = PINNODE2Solver(eq_ode, ode_cfg, bcs)
    # print("\nTreinando PINN (EDO 2ª ordem) para y'' + y = 0, y(0)=0, y'(0)=1 ...")
    # ode_solver.train(verbose_every=500)

    # # Avaliação em uma malha
    # xs = torch.linspace(0.0, math.pi/2, 101)
    # y_pred = ode_solver.predict(xs)
    # # Comparação com sin(x)
    # y_true = torch.sin(xs).cpu()
    # max_abs_err = (y_pred - y_true).abs().max().item()
    # print(f"Erro máximo |y_pred - sin(x)| no domínio: {max_abs_err:.3e}")


    # -----------------------
    # (C) PDE: var4phi = 0
    # -----------------------
    a= 1.0
    b=2.0

    a_t = torch.tensor([[a]], dtype=torch.float32)
    b_t = torch.tensor([[b]], dtype=torch.float32)

    def phi_analytic(r, A, B, C, D):
        return A*torch.log(r) + B*(r**2)*torch.log(r) + C*(r**2) + D
    
    phi_a = phi_analytic(a_t, 2.0,3.0,4.0,5.0).item()    # = 9.0
    phi_b = phi_analytic(b_t, 2.0,3.0,4.0,5.0).item()    # ~30.70406

    # phi' usando autograd ou a expressão analítica:
    def phi_analytic_prime(r, A,B,C,D):
        return A/r + B*(2*r*torch.log(r) + r) + 2*C*r

    phi_p_a = phi_analytic_prime(a_t,2.0,3.0,4.0,5.0).item()  # = 13.0
    phi_p_b = phi_analytic_prime(b_t,2.0,3.0,4.0,5.0).item()  # ~31.31777

    device = "cpu"
    r = lambda x: x          
    ode_params = PDEParams(x=r)
    eq_ode = EquationFactory.create("pde_equation", params=ode_params)

    bcs = [
        DirichletBC(x_b=a, y_b=phi_a),
        DirichletBC(x_b=b, y_b=phi_b),
        NeumannBC(x_b=a, g_b=phi_p_a),
        NeumannBC(x_b=b, g_b=phi_p_b),
        # StressBC(x_b=a, stress_fn=PDEEq.trr, target=0.0),
        # StressBC(x_b=b, stress_fn=PDEEq.ttt, target=0.0),
    ]
    ode_cfg = TrainConfigODE2(
        epochs=12000,
        n_collocation=512,
        lr=1e-3,
        hidden=128,
        depth=6,
        device=device,
        domain=(a, b),
        w_pde=1.0,
        w_bc=0.8,
        normalize_x=True
    )
    ode_solver = PINNODE4Solver(eq_ode, ode_cfg, bcs)

    ode_solver.model._init()
    
    print("\nTreinando PINN (EDO 2ª ordem) para var4phi = 0, Trr(a)=0, Ttt(b)=0 ...")
    ode_solver.train(verbose_every=500)

    # Avaliar rede no domínio
    rs = torch.linspace(a, b, 200).view(-1,1)
    phi_pred = ode_solver.predict(rs)

    # Solução analítica
    def phi_analytic(r, A, B, C, D):
        return A*torch.log(r) + B*(r**2)*torch.log(r) + C*(r**2) + D
    phi_true = phi_analytic(rs, 2, 3, 4, 5)

    # -----------------------
    # Métricas de erro
    # -----------------------
    err = phi_pred - phi_true
    max_err = torch.max(torch.abs(err)).item()
    mse = torch.mean(err**2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(err)).item()
    rel_err = torch.norm(err).item() / torch.norm(phi_true).item()

    print(f"Erro máximo (L∞): {max_err:.6f}")
    print(f"MSE: {mse:.6e}") 
    print(f"RMSE: {rmse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"Erro relativo: {rel_err:.4%}")

    # -----------------------
    # Plot da solução + erros
    # -----------------------
    rs_np = rs.detach().numpy()
    phi_pred_np = phi_pred.detach().numpy()
    phi_true_np = phi_true.detach().numpy()

    print(phi_pred_np)
    print(phi_true_np)
    err_abs = np.abs(phi_pred_np - phi_true_np)
    err_rel = err_abs / np.maximum(np.abs(phi_true_np), 1e-8) * 100

    plt.figure(figsize=(14,5))

    # Curvas PINN vs Analítica
    plt.subplot(1,2,1)
    plt.plot(rs_np, phi_true_np, label="Solução Analítica", color="gold")
    plt.plot(rs_np, phi_pred_np, "--", label="PINN", color="blue")
    plt.xlabel("r")
    plt.ylabel("phi(r)")
    plt.title("Comparação PINN vs Analítica")
    plt.legend()
    plt.grid(True)

    # Erro absoluto e relativo
    plt.subplot(1,2,2)
    plt.plot(rs_np, err_abs, 'r', label='Erro Absoluto')
    # plt.plot(rs_np, err_rel, 'b', label='Erro Relativo (%)')
    plt.xlabel("r")
    plt.ylabel("Erro")
    plt.title("Erros do PINN")
    plt.grid(True)

    plt.tight_layout(pad=2.0)
    plt.show()




if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
