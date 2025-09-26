import math
from typing import List, Any

import torch
import torch.optim as optim

from src.configs.bc import DirichletBC, NeumannBC
from src.configs.train_configs import TrainConfigAlgebraic, TrainConfigODE2
from src.core.equations import QuadraticEquation, ODE2LinearEquation
from src.core.models import MLPBranches, MLP1D


class PINNAlgebraicSolver:
    def __init__(self, equation: QuadraticEquation, cfg: TrainConfigAlgebraic):
        self.eq = equation
        self.cfg = cfg
        self.model = MLPBranches(
            in_dim=1, hidden=cfg.hidden, depth=cfg.depth,
            branches=cfg.branches, out_dim_per_branch=1
        ).to(cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _loss(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        resid = self.eq.residual_algebraic(y, x)
        loss_res = (resid ** 2).mean()
        var_between = y.var(dim=1, unbiased=False).mean()
        loss_div = -var_between
        if self.cfg.w_range > 0:
            y_min, y_max = self.cfg.y_minmax
            below = torch.relu(y_min - y).mean()
            above = torch.relu(y - y_max).mean()
            loss_range = below + above
        else:
            loss_range = torch.tensor(0.0, device=y.device)
        return (self.cfg.w_residual * loss_res +
                self.cfg.w_diversity * loss_div +
                self.cfg.w_range * loss_range)

    def train(self, verbose_every: int = 300):
        device = self.cfg.device
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            t = torch.rand(self.cfg.batch_size, 1, device=device)
            y = self.model(t)
            y = torch.clamp(y, -self.cfg.y_clip, self.cfg.y_clip)
            loss = self._loss(y, t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if epoch % verbose_every == 0 or epoch == 1 or epoch == self.cfg.epochs:
                print(f"[ALG {epoch:05d}] loss={loss.item():.6e}")

    @torch.no_grad()
    def infer(self, n: int = 4096) -> torch.Tensor:
        self.model.eval()
        t = torch.rand(n, 1, device=self.cfg.device)
        y = self.model(t)
        return y.squeeze(-1).mean(dim=0).cpu().numpy()  # média por ramo (ilustrativo)


class PINNODE2Solver:
    def __init__(self,
                 equation: ODE2LinearEquation,
                 cfg: TrainConfigODE2,
                 bcs: List[Any]):
        self.eq = equation
        self.cfg = cfg
        self.bcs = bcs
        self.model = MLP1D(in_dim=1, hidden=cfg.hidden, depth=cfg.depth, out_dim=1).to(cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

        x0, x1 = cfg.domain
        self._a, self._b = x0, x1
        # parâmetros de normalização afim x_norm = (x - a)/(b-a) em [0,1]
        self._scale = 1.0 / (self._b - self._a) if cfg.normalize_x else 1.0
        self._shift = self._a if cfg.normalize_x else 0.0

    def _to_model_in(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize_x:
            return (x - self._shift) * self._scale
        return x

    def _grad(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    def _loss_batch(self) -> torch.Tensor:
        device = self.cfg.device
        # Pontos de colação no domínio original
        xa, xb = self.cfg.domain
        x = torch.rand(self.cfg.n_collocation, 1, device=device) * (xb - xa) + xa
        x.requires_grad_(True)
        x_in = self._to_model_in(x)

        y = self.model(x_in)
        dy_dx_in = self._grad(y, x_in)
        # Se normalizamos x -> x_in, então dy/dx = dy/dx_in * dx_in/dx = dy/dx_in * scale
        scale = self._scale if self.cfg.normalize_x else 1.0
        dy = dy_dx_in * scale
        d2y_dx_in2 = self._grad(dy_dx_in, x_in)
        d2y = d2y_dx_in2 * (scale ** 2)

        # Resíduo PDE
        resid = self.eq.residual_ode2(x, y, dy, d2y)
        loss_pde = (resid ** 2).mean()

        # Perda de BCs
        bc_terms = []
        for bc in self.bcs:
            x_b = torch.tensor([[bc.x_b]], device=device, requires_grad=True)
            x_b_in = self._to_model_in(x_b)
            y_b_pred = self.model(x_b_in)
            if isinstance(bc, DirichletBC):
                bc_terms.append((y_b_pred - bc.y_b) ** 2)
            elif isinstance(bc, NeumannBC):
                dy_b_dx_in = self._grad(y_b_pred, x_b_in)
                dy_b = dy_b_dx_in * scale
                bc_terms.append((dy_b - bc.g_b) ** 2)
            else:
                raise ValueError("Tipo de BC não suportado.")
        loss_bc = torch.stack([t.mean() for t in bc_terms]).mean() if bc_terms else torch.tensor(0.0, device=device)

        return self.cfg.w_pde * loss_pde + self.cfg.w_bc * loss_bc, (loss_pde.item(), loss_bc.item())

    def train(self, verbose_every: int = 500):
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            loss, (lpde, lbc) = self._loss_batch()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if epoch % verbose_every == 0 or epoch == 1 or epoch == self.cfg.epochs:
                print(f"[ODE2 {epoch:05d}] loss={loss.item():.6e} (pde={lpde:.2e}, bc={lbc:.2e})")

    @torch.no_grad()
    def predict(self, x_plot: torch.Tensor) -> torch.Tensor:
        """Retorna y(x_plot) na CPU."""
        self.model.eval()
        x_plot = x_plot.view(-1, 1).to(self.cfg.device)
        x_in = self._to_model_in(x_plot)
        y = self.model(x_in)
        return y.squeeze(-1).cpu()
