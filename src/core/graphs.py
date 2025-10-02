import pandas as pd
import matplotlib.pyplot as plt


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