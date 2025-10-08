import csv
import os
from typing import List, Any

import torch
import torch.optim as optim

from src.configs.bc import DirichletBC, NeumannBC, StressBC
from src.configs.train_configs import TrainConfigODE2
from src.core.equations import AiryStressEquation
from src.core.models import MLP1D

class PINNSolver:
    """
    Resolve uma equação diferencial usando uma Physics-Informed Neural Network (PINN).
    """
    def __init__(
        self,
        equation: AiryStressEquation,
        cfg: TrainConfigODE2,
        bcs: List[Any]
    ):
        self.eq = equation
        self.cfg = cfg
        self.bcs = bcs
        self.model = MLP1D(
            in_dim=1, 
            hidden=cfg.hidden, 
            depth=cfg.depth, 
            out_dim=1
        ).to(cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

        # Parâmetros de normalização do domínio
        x0, x1 = cfg.domain
        self._a, self._b = x0, x1
        self._scale = 1.0 / (self._b - self._a) if cfg.normalize_x else 1.0
        self._shift = self._a if cfg.normalize_x else 0.0

    def _to_model_in(self, x: torch.Tensor) -> torch.Tensor:
        """Normaliza a entrada para o domínio [0, 1] se habilitado."""
        if self.cfg.normalize_x:
            return (x - self._shift) * self._scale
        return x

    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Calcula a derivada de y em relação a x."""
        return torch.autograd.grad(
            y, x, 
            grad_outputs=torch.ones_like(y), 
            create_graph=True, 
            retain_graph=True
        )[0]

    def _loss_batch(self) -> torch.Tensor:
        """Calcula a perda para um lote de pontos de colação e de contorno."""
        device = self.cfg.device
        xa, xb = self.cfg.domain

        # 1. Perda do Resíduo da EDO (PDE Loss)
        x_collocation = torch.rand(self.cfg.n_collocation, 1, device=device) * (xb - xa) + xa
        x_collocation.requires_grad_(True)
        
        x_in = self._to_model_in(x_collocation)
        phi = self.model(x_in)
        
        # O resíduo agora é calculado pela classe da equação, usando a lógica original
        resid = self.eq.residual(phi, x_collocation, x_in, self)
        loss_pde = (resid ** 2).mean()

        # 2. Perda das Condições de Contorno (BC Loss)
        bc_terms = []
        for bc in self.bcs:
            x_b = torch.tensor([[bc.x_b]], device=device, requires_grad=True)
            x_b_in = self._to_model_in(x_b)
            phi_b = self.model(x_b_in)

            if isinstance(bc, StressBC):
                # A função de stress (ex: trr, ttt) é chamada a partir da equação
                # A implementação original de trr usa a coordenada normalizada (x_b_in)
                stress_val = bc.stress_fn(phi_b, x_b_in, self)
                bc_terms.append((stress_val - bc.target) ** 2)
            elif isinstance(bc, DirichletBC):
                bc_terms.append((phi_b - bc.y_b) ** 2)
            elif isinstance(bc, NeumannBC):
                phi_b_r = self._grad(phi_b, x_b)
                bc_terms.append((phi_b_r - bc.g_b) ** 2)
            else:
                raise ValueError(f"Tipo de BC não suportado: {type(bc)}")
        
        loss_bc = torch.stack(bc_terms).mean() if bc_terms else torch.tensor(0.0, device=device)

        return self.cfg.w_pde * loss_pde + self.cfg.w_bc * loss_bc, (loss_pde.item(), loss_bc.item())

    def train(self, verbose_every: int = 500, log_file: str = "training_log.csv"):
        """Executa o loop de treinamento do modelo."""
        self.model.train()
        log_dir = 'docs'
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, log_file), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "pde", "bc"])

            for epoch in range(1, self.cfg.epochs + 1):
                loss, (lpde, lbc) = self._loss_batch()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if epoch % verbose_every == 0 or epoch == 1 or epoch == self.cfg.epochs:
                    writer.writerow([epoch, loss.item(), lpde, lbc])
                    print(f"[PINN {epoch:05d}] loss={loss.item():.6e} (pde={lpde:.2e}, bc={lbc:.2e})")

    @torch.enable_grad()
    def predict_with_stress(self, r_plot: torch.Tensor):
        """
        Avalia o modelo e retorna a função phi e as tensões Trr e Ttt.
        """
        self.model.eval()
        r_plot = r_plot.view(-1, 1).to(self.cfg.device)
        r_plot.requires_grad_(True)
        
        r_in = self._to_model_in(r_plot)
        phi = self.model(r_in)

        # Usa os métodos estáticos da classe da equação para calcular as tensões
        # A implementação original de trr usa a coordenada normalizada (r_in)
        trr_val = self.eq.trr(phi, r_in, self)
        ttt_val = self.eq.ttt(phi, r_plot, self)
        
        momento_val = self.eq.momento(ttt_val, r_plot)

        return (
            phi.detach().squeeze(-1).cpu(),
            trr_val.detach().squeeze(-1).cpu(),
            ttt_val.detach().squeeze(-1).cpu(),
            momento_val
        )
