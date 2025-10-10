import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch


def create_trainning_graph(log_path: str, save_path: str = None):
    df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["loss"], label="Loss total")
    plt.plot(df["epoch"], df["pde"], label="PDE loss")
    plt.plot(df["epoch"], df["bc"], label="BC loss")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Convergência do PINN")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def create_phi_graph(x_coords: torch.Tensor, phi_pred: torch.Tensor, save_path: str = None):
    x_np = x_coords.detach().cpu().numpy()
    phi_pred_np = phi_pred.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x_np, phi_pred_np, "--", label="PINN", color="blue")
    plt.xlabel("r")
    plt.ylabel("phi(r)")
    plt.title("Solução da PINN para phi(r)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def create_radius_graph(rs: any, phi_pred: any, phi_analytic: any = None):
    """Cria um gráfico de phi em função do raio r."""
    rs_np = rs.detach().numpy()
    phi_pred_np = phi_pred.detach().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(rs_np, phi_pred_np, "--", label="PINN", color="blue")

    if phi_analytic is not None:
        phi_analytic_np = phi_analytic.detach().numpy()
        plt.plot(rs_np, phi_analytic_np, label="Analítico", color="red")

    plt.xlabel("Raio (r)")
    plt.ylabel("Phi(r)")
    plt.title("Solução para Phi(r)")
    plt.legend()
    plt.grid(True)
    plt.savefig("docs/phi_vs_radius.png", dpi=300)
    plt.show()
    plt.close()


def create_stress_graph(rs, phi_pred, trr_pred, ttt_pred):
    """Cria um gráfico de phi, Trr e Ttt em função do raio r."""
    rs_np = rs.detach().numpy()
    phi_pred_np = phi_pred.detach().numpy()
    trr_pred_np = trr_pred.detach().numpy()
    ttt_pred_np = ttt_pred.detach().numpy()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo para Phi
    color = 'tab:blue'
    ax1.set_xlabel("Raio (r)")
    ax1.set_ylabel("Phi(r)", color=color)
    ax1.plot(rs_np, phi_pred_np, '--', label="Phi (PINN)", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="--")

    # Eixo para Tensões
    ax2 = ax1.twinx()
    color_trr = 'tab:red'
    color_ttt = 'tab:green'
    ax2.set_ylabel("Tensão", color='black')
    ax2.plot(rs_np, trr_pred_np, label="Trr (PINN)", color=color_trr)
    ax2.plot(rs_np, ttt_pred_np, label="Ttt (PINN)", color=color_ttt)
    ax2.tick_params(axis='y', labelcolor='black')

    # Legendas
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title("Solução para Phi(r) e Tensões (Trr, Ttt)")
    plt.savefig("docs/stress_vs_radius.png", dpi=300)
    plt.show()
    plt.close()

def create_trr_catia():
    df = pd.read_excel('src/core/catia_data.xlsx')
    rs = df["x"]
    trr = df["T"]
    plt.figure(figsize=(8, 5))
    plt.plot(rs, trr, "-", label="Trr catia", color="blue")
    plt.xlabel("Raio (r)")
    plt.ylabel("Trr (N_m2)")
    plt.title("Trr catia em função do Raio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/trr_catia.png", dpi=300)
    plt.show()
    plt.close()


def create_pinn_vs_analytic_report(rs_pinn, trr_analitico, trr_pred):
    """
    Compara Trr Analítico (Verdade) e Predito (PINN), calcula as métricas 
    de erro (MAE e MSE) e gera um gráfico de comparação.

    Args:
        rs_pinn (array-like): Pontos de raio (r).
        trr_analitico (array-like): Valores de Trr da solução Analítica.
        trr_pred (array-like): Valores de Trr preditos pelo PINN.
    """
    print("Iniciando Relatório: PINN vs Analítico...")

    # 1. Preparação e Uniformização dos Dados
    # Converte tensores para numpy e garante que sejam arrays achatados
    rs_pinn_np = np.array(rs_pinn).flatten()
    trr_analitico_np = np.array(trr_analitico).flatten()
    trr_pred_np = np.array(trr_pred).flatten()
    
    # Verifica se os arrays têm o mesmo tamanho
    if not (len(rs_pinn_np) == len(trr_analitico_np) == len(trr_pred_np)):
        print(f"⚠️ Alerta: Os arrays de entrada não têm o mesmo tamanho!")
        return 
        
    # 2. Cálculo das Métricas de Erro (PINN vs Analítico)
    
    # O Analítico é considerado a "verdade" (y_true)
    mae_pinn_vs_analitico = mean_absolute_error(trr_analitico_np, trr_pred_np)
    mse_pinn_vs_analitico = mean_squared_error(trr_analitico_np, trr_pred_np)
    
    metrics = {
        'Métrica': ['MAE (Erro Absoluto Médio)', 'MSE (Erro Quadrático Médio)'],
        'Erro (PINN vs Analítico)': [mae_pinn_vs_analitico, mse_pinn_vs_analitico]
    }
    
    # 3. Criação da Tabela de Erro
    
    df_metrics = pd.DataFrame(metrics).set_index('Métrica')
    print("\n" + "="*50)
    print("Tabela de Erros: Trr PINN vs Trr Analítico")
    print("="*50)
    # Formatação com duas casas decimais em notação científica para clareza
    print(df_metrics.apply(lambda x: pd.Series(["{:.2e}".format(val) for val in x], index=x.index)))
    print("="*50 + "\n")


    # 4. Criação do Gráfico de Comparação
    
    plt.figure(figsize=(10, 6))
    plt.plot(rs_pinn_np, trr_analitico_np, label="Trr Analítico (Verdade)", color="red", linestyle='-', linewidth=2)
    plt.plot(rs_pinn_np, trr_pred_np, label="Trr Predito (PINN)", color="blue", linestyle='--', linewidth=2)
    
    plt.xlabel("Raio (r)")
    plt.ylabel("Trr M(r)")
    plt.title("Comparação: Trr Analítico vs Trr Predito pelo PINN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/trr_pinn_vs_analytic_comparison.png", dpi=300)
    plt.show()
    plt.close()
    
    print("Relatório de PINN vs Analítico concluído.")
    print("Gráfico salvo em docs/trr_pinn_vs_analytic_comparison.png")

#TODO implementar uma unica função para gerar graficos de variavel vs momento
import matplotlib.pyplot as plt
import torch
import numpy as np

def create_trr_analitico_vs_predito_graph(rs, trr_analitico, trr_pred):
    """
    Cria dois gráficos separados do Trr em função do raio r:
    - Gráfico 1: Trr Analítico
    - Gráfico 2: Trr (PINN)
    
    Se os arrays não tiverem o mesmo tamanho, exibe alerta e não plota.
    """

    # Converter tensores para numpy e achatar caso seja 2D
    if isinstance(rs, torch.Tensor):
        rs_np = rs.detach().numpy().flatten()
    else:
        rs_np = np.array(rs).flatten()
        
    if isinstance(trr_analitico, torch.Tensor):
        trr_analitico_np = trr_analitico.detach().numpy().flatten()
    else:
        trr_analitico_np = np.array(trr_analitico).flatten()
        
    if isinstance(trr_pred, torch.Tensor):
        trr_pred_np = trr_pred.detach().numpy().flatten()
    else:
        trr_pred_np = np.array(trr_pred).flatten()
    
    # Verificar se os arrays têm o mesmo tamanho
    if not (len(rs_np) == len(trr_analitico_np) == len(trr_pred_np)):
        print(f"⚠️ Alerta: Os arrays não têm o mesmo tamanho!")
        print(f"rs: {len(rs_np)}, trr_analitico: {len(trr_analitico_np)}, trr_pred: {len(trr_pred_np)}")
        return  # não tenta plotar

    # ---------------- Graph 1: Trr Analítico ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(rs_np, trr_analitico_np, "-", label="Trr Analítico", color="blue")
    plt.xlabel("Raio (r)")
    plt.ylabel("Trr M(r)")
    plt.title("Trr Analítico em função do Raio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/trr_analitico.png", dpi=300)
    plt.show()
    plt.close()

    # ---------------- Graph 2: Trr (PINN) ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(rs_np, trr_pred_np, "--", label="Trr (PINN)", color="purple")
    plt.xlabel("Raio (r)")
    plt.ylabel("Trr M(r)")
    plt.title("Trr (PINN) em função do Raio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/trr_predito.png", dpi=300)
    plt.show()
    plt.close()

