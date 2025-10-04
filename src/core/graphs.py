import pandas as pd
import matplotlib.pyplot as plt
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


def create_stress_and_moment_graph(r_coords: torch.Tensor, trr: torch.Tensor, ttt: torch.Tensor, moment: torch.Tensor, save_path: str = None):
    r_np = r_coords.detach().cpu().numpy()
    trr_np = trr.detach().cpu().numpy()
    ttt_np = ttt.detach().cpu().numpy()
    moment_np = moment.detach().cpu().numpy()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo para Tensões
    ax1.set_xlabel("Raio (r)")
    ax1.set_ylabel("Tensão", color="tab:red")
    ax1.plot(r_np, trr_np, label="Tensão Radial (Trr)", color="red", linestyle='--')
    ax1.plot(r_np, ttt_np, label="Tensão Tangencial (Ttt)", color="green", linestyle=':')
    ax1.tick_params(axis='y', labelcolor="tab:red")
    ax1.grid(True, which="both", ls="--")

    # Eixo para Momento
    ax2 = ax1.twinx()
    ax2.set_ylabel("Momento (M)", color="tab:blue")
    ax2.plot(r_np, moment_np, label="Momento (M)", color="blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    # Legendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Distribuição de Tensões e Momento ao Longo do Raio")
    fig.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def create_trr_graph(r_coords: torch.Tensor, trr: torch.Tensor, save_path: str = None):
    r_np = r_coords.detach().cpu().numpy()
    trr_np = trr.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(r_np, trr_np, label="Tensão Radial (Trr)", color="red")
    plt.xlabel("Raio (r)")
    plt.ylabel("Tensão Radial (Trr)")
    plt.title("Tensão Radial vs. Raio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=2.0)
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
