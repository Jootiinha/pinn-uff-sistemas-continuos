import random
import torch
import yaml
from pathlib import Path

from src.configs.bc import StressBC
from src.configs.train_configs import TrainConfigODE2
from src.core.equations import EquationFactory, AiryStressEquation
from src.core.solvers import PINNSolver
from src.core import graphs
from src.experiments.pde_4th_order import config as cfg

def load_config(config_path):
    """Carrega as configurações do arquivo YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_experiment(config):
    """Configura e retorna o solver PINN com base no arquivo de configuração."""
    domain_cfg = config['domain']
    train_cfg = config['training']

    a = domain_cfg['a']
    b = domain_cfg['b']

    # Usa a nova classe de equação refatorada
    equation = EquationFactory.create("airy_stress")

    bcs = [
        StressBC(x_b=a, stress_fn=AiryStressEquation.trr, target=0.0),
        StressBC(x_b=b, stress_fn=AiryStressEquation.ttt, target=0.0),
    ]
    
    ode_cfg = TrainConfigODE2(
        epochs=train_cfg['epochs'],
        n_collocation=train_cfg['n_collocation'],
        lr=train_cfg['learning_rate'],
        hidden=train_cfg['hidden_units'],
        depth=train_cfg['network_depth'],
        device=train_cfg['device'],
        domain=(a, b),
        w_pde=train_cfg['weight_pde'],
        w_bc=train_cfg['weight_bc'],
        normalize_x=train_cfg['normalize_x']
    )
    
    # Usa o novo solver refatorado
    return PINNSolver(equation, ode_cfg, bcs)

def train_model(solver):
    """Treina o modelo PINN."""
    print("\nIniciando treinamento do modelo PINN...")
    solver.train(verbose_every=500)
    graphs.create_trainning_graph()

def evaluate_and_plot(solver, config):
    """Avalia o modelo treinado e gera os gráficos de resultado."""
    domain_cfg = config['domain']
    eval_cfg = config['evaluation']
    
    a = domain_cfg['a']
    b = domain_cfg['b']
    moment_scale_factor = eval_cfg['moment_scale_factor']

    rs = torch.linspace(a, b, 200).view(-1, 1)

    phi_pred, trr_pred, ttt_pred, momento_val = solver.predict_with_stress(rs)

    trr_pred_M = trr_pred * moment_scale_factor * momento_val
    trr_analitico = AiryStressEquation.T_rr_analytical(rs, a, b, momento_val)

    graphs.create_radius_graph(rs, phi_pred)
    graphs.create_stress_graph(rs, phi_pred, trr_pred, ttt_pred)
    graphs.create_trr_analitico_vs_predito_graph(rs, trr_analitico, trr_pred_M)
    graphs.create_trr_catia()

    graphs.create_pinn_vs_analytic_report(
        rs_pinn=rs, 
        trr_analitico=trr_analitico, 
        trr_pred=trr_pred_M
    )

def main():
    """Ponto de entrada principal da aplicação."""
    # Define o caminho do arquivo de configuração relativo ao script atual
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    
    pinn_solver = setup_experiment(config)

    # A chamada explícita a _init() é crucial para garantir a reprodutibilidade
    # exata do resultado, pois reseta os pesos do modelo para o mesmo estado
    # inicial em todas as execuções, em conjunto com a seed.
    if hasattr(pinn_solver.model, '_init'):
        pinn_solver.model._init()
    
    train_model(pinn_solver)
    evaluate_and_plot(pinn_solver, config)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
