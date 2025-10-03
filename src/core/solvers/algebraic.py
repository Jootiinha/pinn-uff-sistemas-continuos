import torch
import torch.optim as optim

from src.configs.train_configs import TrainConfigAlgebraic
from src.core.equations.quadratic import QuadraticEquation
from src.core.models import MLPBranches


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
