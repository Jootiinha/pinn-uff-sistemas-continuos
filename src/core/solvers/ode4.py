import inspect
from typing import List, Any

import torch

from src.configs.bc import DirichletBC, NeumannBC, StressBC
from src.configs.train_configs import TrainConfigODE4
from src.core.equations.ode4_elasticity import ODE4thOrderEquation
from src.core.solvers.base import PINNSolver


class PINNODE4Solver(PINNSolver):
    def __init__(self, equation: ODE4thOrderEquation, cfg: TrainConfigODE4, bcs: List[Any]):
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
