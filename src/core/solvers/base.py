import os
import csv
import torch
import torch.optim as optim

from src.core.models import MLP1D
from src.configs.train_configs import TrainConfigODE2, TrainConfigODE4


class PINNSolver:
    """Classe base para solvers de PINN para EDOs."""
    def __init__(self, equation, cfg, bcs):
        self.eq = equation
        self.cfg = cfg
        self.bcs = bcs
        self.model = MLP1D(in_dim=1, hidden=cfg.hidden, depth=cfg.depth, out_dim=1).to(cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

        x0, x1 = cfg.domain
        self._a, self._b = x0, x1
        self._scale = 1.0 / (self._b - self._a) if cfg.normalize_x else 1.0
        self._shift = self._a if cfg.normalize_x else 0.0

    def _to_model_in(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._shift) * self._scale if self.cfg.normalize_x else x

    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    def _loss_batch(self) -> tuple[torch.Tensor, tuple[float, float]]:
        raise NotImplementedError("Subclasses devem implementar _loss_batch.")

    def train(self, verbose_every: int = 500, log_file: str = "training_log.csv"):
        self.model.train()
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "pde", "bc"])

            for epoch in range(1, self.cfg.epochs + 1):
                loss, (lpde, lbc) = self._loss_batch()
                self.opt.zero_grad()
                loss.backward()
                
                # Esta é a linha que efetivamente atualiza os pesos do modelo__
                #   , utilizando o otimizador (neste caso, Adam) 
                #   para ajustar os parâmetros com base nos gradientes calculados.
                self.opt.step()

                if epoch % verbose_every == 0 or epoch == 1 or epoch == self.cfg.epochs:
                    writer.writerow([epoch, loss.item(), lpde, lbc])
                    print(f"[PINN {epoch:05d}] loss={loss.item():.6e} (pde={lpde:.2e}, bc={lbc:.2e})")

    @torch.no_grad()
    def predict(self, x_plot: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x_plot = x_plot.view(-1, 1).to(self.cfg.device)
        x_in = self._to_model_in(x_plot)
        y = self.model(x_in)
        return y.squeeze(-1).cpu()

    def save_model(self, path: str):
        """Salva o estado do modelo em um arquivo."""
        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Modelo salvo em {path}")

    def load_model(self, path: str):
        """Carrega o estado do modelo de um arquivo."""
        self.model.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.model.eval()
        print(f"Modelo carregado de {path}")
