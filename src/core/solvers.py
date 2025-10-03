import math
import csv
import os
import inspect
from typing import List, Any

import torch
import torch.optim as optim

from src.configs.bc import DirichletBC, NeumannBC, StressBC
from src.configs.train_configs import TrainConfigAlgebraic, TrainConfigODE2, TrainConfigODE4
from src.core.equations.ode2_linear import ODE2LinearEquation
from src.core.equations.ode4_elasticity import ODE4thOrderEquation
from src.core.equations.quadratic import QuadraticEquation
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


class PINNODE2Solver(PINNSolver):
    def __init__(self, equation: ODE2LinearEquation, cfg: TrainConfigODE2, bcs: List[Any]):
        super().__init__(equation, cfg, bcs)

    def _loss_batch(self) -> tuple[torch.Tensor, tuple[float, float]]:
        device = self.cfg.device
        xa, xb = self.cfg.domain
        x = torch.rand(self.cfg.n_collocation, 1, device=device) * (xb - xa) + xa
        x.requires_grad_(True)
        x_in = self._to_model_in(x)

        y = self.model(x_in)
        dy_dx_in = self._grad(y, x_in)
        dy = dy_dx_in * self._scale
        d2y_dx_in2 = self._grad(dy_dx_in, x_in)
        d2y = d2y_dx_in2 * (self._scale ** 2)

        resid = self.eq.residual_ode2(x, y, dy, d2y)
        loss_pde = (resid ** 2).mean()

        bc_terms = []
        for bc in self.bcs:
            x_b = torch.tensor([[bc.x_b]], device=device, requires_grad=True)
            x_b_in = self._to_model_in(x_b)
            y_b = self.model(x_b_in)
            if isinstance(bc, DirichletBC):
                bc_terms.append((y_b - bc.y_b) ** 2)
            elif isinstance(bc, NeumannBC):
                dy_b_dx_in = self._grad(y_b, x_b_in)
                dy_b = dy_b_dx_in * self._scale
                bc_terms.append((dy_b - bc.g_b) ** 2)
            else:
                raise ValueError("Tipo de BC não suportado.")
        loss_bc = torch.stack([t.mean() for t in bc_terms]).mean() if bc_terms else torch.tensor(0.0, device=device)

        total_loss = self.cfg.w_pde * loss_pde + self.cfg.w_bc * loss_bc
        return total_loss, (loss_pde.item(), loss_bc.item())


class PINNODE4Solver(PINNSolver):
    def __init__(self, equation: ODE4thOrderEquation, cfg: TrainConfigODE4, bcs: List[Any]):
        super().__init__(equation, cfg, bcs)

    def _loss_batch(self) -> tuple[torch.Tensor, tuple[float, float]]:
        device = self.cfg.device
        xa, xb = self.cfg.domain
        x = torch.rand(self.cfg.n_collocation, 1, device=device) * (xb - xa) + xa
        x.requires_grad_(True)
        x_in = self._to_model_in(x)

        y = self.model(x_in)
        dy_dx_in = self._grad(y, x_in)
        dy = dy_dx_in * self._scale
        d2y_dx_in2 = self._grad(dy_dx_in, x_in)
        d2y = d2y_dx_in2 * (self._scale ** 2)
        d3y_dx_in3 = self._grad(d2y_dx_in2, x_in)
        d3y = d3y_dx_in3 * (self._scale ** 3)
        d4y_dx_in4 = self._grad(d3y_dx_in3, x_in)
        d4y = d4y_dx_in4 * (self._scale ** 4)

        resid = self.eq.residual_ode4(x, d2y, d3y, d4y)
        loss_pde = (resid ** 2).mean()

        bc_terms = []
        for bc in self.bcs:
            x_b = torch.tensor([[bc.x_b]], device=device, requires_grad=True)
            x_b_in = self._to_model_in(x_b)
            y_b = self.model(x_b_in)

            if isinstance(bc, StressBC):
                dy_b_dx_in = self._grad(y_b, x_b_in)
                dy_b = dy_b_dx_in * self._scale
                d2y_b_dx_in2 = self._grad(dy_b_dx_in, x_b_in)
                d2y_b = d2y_b_dx_in2 * (self._scale ** 2)

                # Inspeciona a assinatura da função de stress para passar apenas os argumentos necessários
                sig = inspect.signature(bc.stress_fn)
                kwargs = {'x': x_b, 'y': y_b, 'dy': dy_b, 'd2y': d2y_b}
                pass_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                
                stress_val = bc.stress_fn(**pass_kwargs)
                bc_terms.append((stress_val - bc.target) ** 2)
            elif isinstance(bc, DirichletBC):
                bc_terms.append((y_b - bc.y_b) ** 2)
            elif isinstance(bc, NeumannBC):
                dy_b_dx_in = self._grad(y_b, x_b_in)
                dy_b = dy_b_dx_in * self._scale
                bc_terms.append((dy_b - bc.g_b) ** 2)
            else:
                raise ValueError("Tipo de BC não suportado.")
        
        loss_bc = torch.stack([t.mean() for t in bc_terms]).mean() if bc_terms else torch.tensor(0.0, device=device)

        total_loss = self.cfg.w_pde * loss_pde + self.cfg.w_bc * loss_bc
        return total_loss, (loss_pde.item(), loss_bc.item())
