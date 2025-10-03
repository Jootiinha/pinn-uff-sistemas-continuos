import torch
import random

from src.core.solvers import PINNODE4Solver
from src.core import graphs
from src.experiments.pde_4th_order import config as cfg

def main():
    """
    Carrega uma PINN pré-treinada para resolver a equação de 4ª ordem
    e faz previsões.
    """
    # 1. Instanciar o solver (a arquitetura deve ser a mesma do modelo salvo)
    ode_solver = PINNODE4Solver(cfg.equation, cfg.train_config, cfg.bcs)

    # 2. Carregar o modelo treinado
    model_path = "docs/pde_4th_order_model.pth"
    ode_solver.load_model(model_path)

    print(f"Modelo carregado de '{model_path}'. Fazendo previsões...")

    # 3. Avaliar a rede no domínio (fazer previsões)
    rs = torch.linspace(cfg.a, cfg.b, 200).view(-1, 1)
    phi_pred = ode_solver.predict(rs)

    # 4. Gerar gráfico da predição
    output_graph_path = "docs/pde_4th_order_phi_from_loaded_model.png"
    graphs.create_phi_graph(rs, phi_pred, output_graph_path)
    print(f"Gráfico da predição salvo em '{output_graph_path}'")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
