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
