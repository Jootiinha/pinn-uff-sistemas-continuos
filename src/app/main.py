import torch
import random
import argparse
import os

from src.core.solvers import PINNODE4Solver
from src.core import graphs
from src.configs.loader import load_config
from src.configs.bc import MomentBC

def main():
    """
    Treina uma PINN para resolver uma equação diferencial com base em um arquivo
    de configuração YAML.
    """

    parser = argparse.ArgumentParser(description="Treinar uma PINN usando um arquivo de configuração YAML.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Caminho para o arquivo de configuração do experimento (ex: src/experiments/pde_4th_order/experiment.yaml)"
    )
    args = parser.parse_args()

    print(f"Carregando configuração de: {args.config_path}")
    config = load_config(args.config_path)
    
    experiment_name = config["experiment_name"]
    equation = config["equation"]
    train_config = config["train_config"]
    bcs = config["bcs"]
    a, b = config["domain"]

    ode_solver = PINNODE4Solver(equation, train_config, bcs)
    
    print(f"\nIniciando experimento: {experiment_name}")
    print("Treinando PINN...")

    output_dir = "docs"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{experiment_name}_log.csv")
    model_path = os.path.join(output_dir, f"{experiment_name}_model.pth")
    metrics_graph_path = os.path.join(output_dir, f"{experiment_name}_metrics.png")
    phi_graph_path = os.path.join(output_dir, f"{experiment_name}_phi.png")

    ode_solver.train(verbose_every=500, log_file=log_path)

    ode_solver.save_model(model_path)
    print(f"Modelo salvo em: {model_path}")
    
    graphs.create_trainning_graph(log_path, metrics_graph_path)
    print(f"Gráfico de métricas salvo em: {metrics_graph_path}")

    rs = torch.linspace(a, b, 200).view(-1, 1)
    phi_pred = ode_solver.predict(rs)
    graphs.create_phi_graph(rs, phi_pred, phi_graph_path)
    print(f"Gráfico da solução salvo em: {phi_graph_path}")

    # Calcular e plotar as tensões e o momento
    stress_moment_graph_path = os.path.join(output_dir, f"{experiment_name}_stresses_and_moment.png")
    rs.requires_grad_(True)
    phi = ode_solver.model(rs)
    d_phi_dr = ode_solver._grad(phi, rs)
    d2_phi_dr2 = ode_solver._grad(d_phi_dr, rs)
    
    trr = ode_solver.eq.trr(rs, d_phi_dr)
    ttt = ode_solver.eq.ttt(rs, d2_phi_dr2)
    moment = ode_solver.eq.moment_r(rs, d2_phi_dr2)
    
    graphs.create_stress_and_moment_graph(rs, trr, ttt, moment, stress_moment_graph_path)
    print(f"Gráfico de tensões e momento salvo em: {stress_moment_graph_path}")

    # Gerar e salvar o gráfico de Tensão Radial (Trr)
    trr_graph_path = os.path.join(output_dir, f"{experiment_name}_trr.png")
    graphs.create_trr_graph(rs, trr, trr_graph_path)
    print(f"Gráfico de Tensão Radial salvo em: {trr_graph_path}")

    target_moment = "N/A"
    for bc in bcs:
        if isinstance(bc, MomentBC):
            target_moment = bc.target
            break
    print(f"Momento Alvo: {target_moment}")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
