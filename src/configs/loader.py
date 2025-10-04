import yaml
from typing import Any, Dict, List

from src.configs.bc import StressBC, MomentBC
from src.configs.train_configs import TrainConfigODE4
from src.core.equations.factory import EquationFactory

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega a configuração de um arquivo YAML e instancia os objetos necessários.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Carregar a equação
    equation_name = config['equation']['name']
    equation = EquationFactory.create(equation_name)

    # Carregar configuração de treino
    train_cfg_data = config['training']
    domain_cfg = config['domain']
    train_config = TrainConfigODE4(
        epochs=train_cfg_data['epochs'],
        n_collocation=train_cfg_data['n_collocation'],
        lr=train_cfg_data['learning_rate'],
        hidden=train_cfg_data['model']['hidden'],
        depth=train_cfg_data['model']['depth'],
        device=config['device'],
        domain=(domain_cfg['a'], domain_cfg['b']),
        w_pde=train_cfg_data['weights']['pde'],
        w_bc=train_cfg_data['weights']['bc'],
        normalize_x=train_cfg_data['normalize_x'],
        y_a=train_cfg_data['dirichlet_bcs'].get('y_a'),
        y_b=train_cfg_data['dirichlet_bcs'].get('y_b'),
    )

    # Carregar condições de contorno
    bcs = []
    for bc_data in config['boundary_conditions']:
        bc_type = bc_data['type']
        
        if bc_type == "StressBC":
            point_key = bc_data['point']
            point_val = domain_cfg[point_key]
            stress_fn_name = bc_data['stress_function']
            stress_fn = getattr(equation, stress_fn_name)
            bcs.append(StressBC(
                x_b=point_val,
                stress_fn=stress_fn,
                target=bc_data['target']
            ))
        
        elif bc_type == "MomentBC":
            moment_fn_name = bc_data['moment_function']
            moment_fn = getattr(equation, moment_fn_name)
            bcs.append(MomentBC(
                moment_fn=moment_fn,
                target=bc_data['target']
            ))
            
        # Adicionar outros tipos de BC aqui se necessário

    return {
        "experiment_name": config['experiment_name'],
        "equation": equation,
        "train_config": train_config,
        "bcs": bcs,
        "domain": (domain_cfg['a'], domain_cfg['b'])
    }
