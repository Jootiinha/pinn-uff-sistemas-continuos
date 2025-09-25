# pinn_factory_quadratic_and_ode2.py
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim


class BaseEquation:
    """Interface/base para equações.
    O solver pode chamar .residual(...) com assinaturas diferentes.
    Para manter compatibilidade, aceitamos **kwargs.
    """
    def solution_dim(self) -> int:
        return 1

    # Para problemas ALGÉBRICOS (ex.: encontrar raízes), usamos y (constante)
    def residual_algebraic(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # Para EDO de 2ª ordem, usamos x, y(x), y'(x), y''(x)
    def residual_ode2(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ------------------------------
# 1.a) Equação algébrica: 2º grau
# ------------------------------

@dataclass
class QuadraticParams:
    a: float
    b: float
    c: float

class QuadraticEquation(BaseEquation):
    """a*y^2 + b*y + c = 0 (tratando o desconhecido como y)."""
    def __init__(self, params: QuadraticParams):
        self.a = float(params.a)
        self.b = float(params.b)
        self.c = float(params.c)

    def residual_algebraic(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.a * y**2 + self.b * y + self.c

    def analytic_roots(self) -> Tuple[complex, complex]:
        a, b, c = self.a, self.b, self.c
        if abs(a) < 1e-15:
            if abs(b) < 1e-15:  # degenerado
                return (complex("nan"), complex("nan"))
            r = -c / b
            return (complex(r), complex("nan"))
        disc = b*b - 4*a*c
        if disc >= 0:
            r1 = (-b + math.sqrt(disc)) / (2*a)
            r2 = (-b - math.sqrt(disc)) / (2*a)
            return (complex(r1), complex(r2))
        else:
            real = -b / (2*a)
            imag = math.sqrt(-disc) / (2*a)
            return (complex(real, imag), complex(real, -imag))


# ------------------------------
# 1.b) EDO 2ª ordem linear geral
# y'' + p(x) y' + q(x) y = r(x)
# ------------------------------

@dataclass
class ODE2LinearParams:
    p: Callable[[torch.Tensor], torch.Tensor]  # p(x)
    q: Callable[[torch.Tensor], torch.Tensor]  # q(x)
    r: Callable[[torch.Tensor], torch.Tensor]  # r(x)

class ODE2LinearEquation(BaseEquation):
    def __init__(self, params: ODE2LinearParams):
        self.p_fun = params.p
        self.q_fun = params.q
        self.r_fun = params.r

    def residual_ode2(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        # todos como tensores (N, 1)
        p = self.p_fun(x)
        q = self.q_fun(x)
        r = self.r_fun(x)
        return d2y + p * dy + q * y - r


# ------------------------------
# Factory
# ------------------------------

class EquationFactory:
    _REGISTRY: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, ctor):
        cls._REGISTRY[name] = ctor

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEquation:
        if name not in cls._REGISTRY:
            raise ValueError(f"Equação '{name}' não registrada.")
        return cls._REGISTRY[name](**kwargs)


EquationFactory.register("quadratic", lambda params: QuadraticEquation(params=params))
EquationFactory.register("ode2_linear", lambda params: ODE2LinearEquation(params=params))


# =============================================================================
# 2) MODELOS MLP
# =============================================================================

class MLPBranches(nn.Module):
    """MLP para problema ALGÉBRICO: recebe t (dummy) e retorna K saídas (ramos)."""
    def __init__(self, in_dim=1, hidden=64, depth=3, branches=2, out_dim_per_branch=1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.Tanh()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(dim, branches * out_dim_per_branch)
        self.branches = branches
        self.out_dim_per_branch = out_dim_per_branch
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.net(t)
        y = self.head(h)
        return y.view(-1, self.branches, self.out_dim_per_branch)


class MLP1D(nn.Module):
    """MLP para EDO: recebe x e retorna y(x)."""
    def __init__(self, in_dim=1, hidden=64, depth=4, out_dim=1, act=nn.Tanh):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), act()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(dim, out_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# =============================================================================
# 3) SOLVER ALGÉBRICO (raízes de 2º grau) – igual ao anterior, resumido
# =============================================================================

@dataclass
class TrainConfigAlgebraic:
    epochs: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    branches: int = 2
    hidden: int = 64
    depth: int = 3
    device: str = "cpu"
    y_clip: float = 1e6
    w_residual: float = 1.0
    w_diversity: float = 0.02
    w_range: float = 0.0
    y_minmax: Tuple[float, float] = (-1e3, 1e3)

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


# =============================================================================
# 4) SOLVER EDO 2ª ORDEM
# =============================================================================

@dataclass
class TrainConfigODE2:
    epochs: int = 6000
    n_collocation: int = 256
    lr: float = 1e-3
    hidden: int = 64
    depth: int = 4
    device: str = "cpu"
    domain: Tuple[float, float] = (0.0, math.pi/2)  # [x0, x1]
    w_pde: float = 1.0
    w_bc: float = 1.0
    # opcional: normalização do domínio para estabilidade
    normalize_x: bool = True

# Representação de BCs simples:
# - Dirichlet: y(x_b) = y_b
# - Neumann:   y'(x_b) = g_b
@dataclass
class DirichletBC:
    x_b: float
    y_b: float

@dataclass
class NeumannBC:
    x_b: float
    g_b: float  # valor de y'(x_b)

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


# =============================================================================
# 5) EXEMPLO DE USO
# =============================================================================

def main():
    # -----------------------
    # (A) Algébrico: x^2 - 5x + 6 = 0  -> raízes 2 e 3
    # -----------------------
    quad_params = QuadraticParams(a=1.0, b=-5.0, c=6.0)
    eq_quad = EquationFactory.create("quadratic", params=quad_params)
    alg_cfg = TrainConfigAlgebraic(epochs=2000, device="cpu")
    alg_solver = PINNAlgebraicSolver(eq_quad, alg_cfg)
    print("Treinando PINN (algébrico) para raízes de 2º grau...")
    alg_solver.train(verbose_every=400)
    print("Raízes analíticas:", eq_quad.analytic_roots())

    # -----------------------
    # (B) EDO 2ª ordem: y'' + y = 0
    # Condições: y(0)=0 (Dirichlet) e y'(0)=1 (Neumann) -> y = sin(x)
    # Domínio: [0, pi/2]
    # -----------------------
    device = "cpu"
    p = lambda x: torch.zeros_like(x)         # p(x)=0
    q = lambda x: torch.ones_like(x)          # q(x)=1
    r = lambda x: torch.zeros_like(x)         # r(x)=0
    ode_params = ODE2LinearParams(p=p, q=q, r=r)
    eq_ode = EquationFactory.create("ode2_linear", params=ode_params)

    bcs = [
        DirichletBC(x_b=0.0, y_b=0.0),  # y(0)=0
        NeumannBC(x_b=0.0, g_b=1.0),    # y'(0)=1
    ]
    ode_cfg = TrainConfigODE2(
        epochs=6000,
        n_collocation=256,
        lr=1e-3,
        hidden=64,
        depth=4,
        device=device,
        domain=(0.0, math.pi/2),
        w_pde=1.0,
        w_bc=1.0,
        normalize_x=True
    )
    ode_solver = PINNODE2Solver(eq_ode, ode_cfg, bcs)
    print("\nTreinando PINN (EDO 2ª ordem) para y'' + y = 0, y(0)=0, y'(0)=1 ...")
    ode_solver.train(verbose_every=500)

    # Avaliação em uma malha
    xs = torch.linspace(0.0, math.pi/2, 101)
    y_pred = ode_solver.predict(xs)
    # Comparação com sin(x)
    y_true = torch.sin(xs).cpu()
    max_abs_err = (y_pred - y_true).abs().max().item()
    print(f"Erro máximo |y_pred - sin(x)| no domínio: {max_abs_err:.3e}")

    # Dica: plote xs vs y_pred para inspecionar a solução.


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    main()
