from typing import List, Any

import torch

from src.configs.bc import DirichletBC, NeumannBC
from src.configs.train_configs import TrainConfigODE2
from src.core.equations.ode2_linear import ODE2LinearEquation
from src.core.solvers.base import PINNSolver


class PINNODE2Solver(PINNSolver):
    def __init__(self, equation: ODE2LinearEquation, cfg: TrainConfigODE2, bcs: List[Any]):
        super().__init__(equation, cfg, bcs)
        
        # Adiciona BCs de Dirichlet se y_a ou y_b forem especificados
        xa, xb = self.cfg.domain
        if cfg.y_a is not None:
            self.bcs.append(DirichletBC(x_b=xa, y_b=cfg.y_a))
        if cfg.y_b is not None:
            self.bcs.append(DirichletBC(x_b=xb, y_b=cfg.y_b))

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
