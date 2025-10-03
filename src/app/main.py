import torch
import random

from src.core.solvers import PINNODE4Solver
from src.core import graphs
from src.experiments.pde_4th_order import config as cfg

def main():
    """
    Treina uma PINN para resolver a equação de 4ª ordem definida em
    'experiments/pde_4th_order/config.py'.
    """
    # Solver
    ode_solver = PINNODE4Solver(cfg.equation, cfg.train_config, cfg.bcs)
    
    print("\nTreinando PINN (EDO 4ª ordem)...")
    log_path = "docs/pde_4th_order_log.csv"
    ode_solver.train(verbose_every=500, log_file=log_path)
    
    # Gráficos
    graphs.create_trainning_graph(log_path, "docs/pde_4th_order_metrics.png")

    # Avaliar rede no domínio
    rs = torch.linspace(cfg.a, cfg.b, 200).view(-1,1)
    phi_pred = ode_solver.predict(rs)
    graphs.create_phi_graph(rs, phi_pred, "docs/pde_4th_order_phi.png")

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
