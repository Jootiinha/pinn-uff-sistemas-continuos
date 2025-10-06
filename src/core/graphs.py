import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_trainning_graph():

    df = pd.read_csv("docs/training_log.csv")

    plt.figure(figsize=(8,5))
    plt.plot(df["epoch"], df["loss"], label="Loss total")
    plt.plot(df["epoch"], df["pde"], label="PDE loss")
    plt.plot(df["epoch"], df["bc"], label="BC loss")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Convergência do PINN")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("docs/training_metrics.png", dpi=300)
    plt.show()
    plt.close()


def create_phi_graph(phi_pred: any):

    phi_pred_np = phi_pred.detach().numpy()

    # # Curvas PINN 
    plt.subplot(1,2,1)
    plt.plot(phi_pred_np, "--", label="PINN", color="blue")
    plt.grid(True)
    plt.tight_layout(pad=2.0)
    plt.savefig("docs/phi_predict.png", dpi=300)
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


# # def create_moment_graph(rs, moment):
#     """Cria um gráfico do momento em função do raio r, calculado como a integral de r * Ttt(r)."""
#     rs_np = rs.detach().numpy().flatten()
#     ttt_pred_np = ttt_pred.detach().numpy().flatten()

#     # Calcula o integrando: r * Ttt(r)
#     integrand = rs_np * ttt_pred_np

#     # Calcula a integral cumulativa usando a regra do trapézio
#     # M(r) = ∫[de b a r] (x * Ttt(x)) dx, onde 'b' é o primeiro valor em rs_np
#     dr = rs_np[1] - rs_np[0]  # Assumindo espaçamento uniforme
    
#     # Usamos np.cumsum para uma aproximação da integral cumulativa
#     cumulative_integral = np.cumsum((integrand[:-1] + integrand[1:]) / 2) * dr
    
#     # Adiciona um zero no início para que o array tenha o mesmo tamanho de rs_np
#     moment = np.insert(cumulative_integral, 0, 0)

#     plt.figure(figsize=(8, 5))
#     plt.plot(rs_np, moment, "--", label="Momento (PINN)", color="purple")
#     plt.xlabel("Raio (r)")
#     plt.ylabel("Momento M(r)")
#     plt.title("Momento em função do Raio")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("docs/moment_vs_radius.png", dpi=300)
#     plt.show()
#     plt.close()


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

