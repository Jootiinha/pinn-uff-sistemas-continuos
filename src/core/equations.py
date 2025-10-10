import math
import numpy as np
import torch
from scipy import integrate
from typing import Dict, Any

class BaseEquation:
    """Classe base para equações a serem resolvidas pela PINN."""
    def solution_dim(self) -> int:
        return 1

class AiryStressEquation(BaseEquation):
    """
    Representa a equação diferencial de 4ª ordem para a função de tensão de Airy (phi)
    em coordenadas polares, usada para resolver problemas de elasticidade 2D.

    A equação é: ∇⁴φ = 0, que em coordenadas polares (assumindo simetria axial)
    se simplifica para uma EDO de 4ª ordem em r.
    """

    def residual(self, phi: torch.Tensor, r: torch.Tensor, r_in: torch.Tensor, solver) -> torch.Tensor:
        """
        Calcula o resíduo da EDO de 4ª ordem.
        Esta é a implementação original do projeto, que é mantida para garantir
        a reprodutibilidade dos resultados.
        """
        # Replicando a lógica original do PINNODE4Solver
        dy_dx_in = solver._grad(phi, r_in)
        dy = dy_dx_in
        
        d2y_dx_in2 = solver._grad(dy_dx_in, r_in)
        d2y = d2y_dx_in2

        g = d2y + dy
        dg_dx_x = solver._grad(g / r, r)
        dg_dx = solver._grad(g, r)
        d2g_dx = solver._grad(dg_dx, r)

        # Resíduo da EDO 4° ordem original
        return dg_dx_x * d2g_dx

    @staticmethod
    def trr(phi: torch.Tensor, r: torch.Tensor, solver) -> torch.Tensor:
        """
        Calcula a tensão radial (Trr).
        Esta é a implementação original do projeto, que é mantida para garantir
        a reprodutibilidade dos resultados.
        """
        # A implementação original era d/dr(phi/r), que é diferente de (1/r)d(phi)/dr.
        phi_r = solver._grad(phi / r, r)
        return phi_r

    @staticmethod
    def ttt(phi: torch.Tensor, r: torch.Tensor, solver) -> torch.Tensor:
        """Calcula a tensão tangencial (Ttt) = d^2(phi)/dr^2."""
        phi_r = solver._grad(phi, r)
        phi_rr = solver._grad(phi_r, r)
        return phi_rr

    @staticmethod
    def momento(ttt_val: torch.Tensor, r_in: torch.Tensor) -> float:
        """Calcula o momento fletor M integrando Ttt * r."""
        ttt_np = ttt_val.detach().cpu().numpy().flatten()
        r_in_np = r_in.detach().cpu().numpy().flatten()
        
        # Garante que os dados estão ordenados pelo raio para a integração
        sorted_indices = np.argsort(r_in_np)
        r_in_sorted = r_in_np[sorted_indices]
        ttt_sorted = ttt_np[sorted_indices]

        integrando = ttt_sorted * r_in_sorted
        M = integrate.simpson(integrando, r_in_sorted)
        return float(M)

    @staticmethod
    def T_rr_analytical(r: torch.Tensor, a: float, b: float, M: float) -> torch.Tensor:
        """Calcula a solução analítica para a tensão radial Trr."""
        r = r.flatten()
        
        # Constantes para simplificar a fórmula
        a2 = a**2
        b2 = b**2
        log_b_a = torch.log(torch.tensor(b / a, dtype=torch.float32))

        # Termos da equação
        term1 = ((a2 * b2) / r**2) * log_b_a
        term2 = b2 * torch.log(r / b)
        term3 = a2 * torch.log(a / r)
        
        # Denominador da fração
        denominator = (b2 - a2)**2 - 4 * a2 * b2 * (log_b_a**2)
        
        # Evita divisão por zero
        if abs(denominator) < 1e-9:
            return torch.full_like(r, float('nan'))

        Trr = (4 * M / denominator) * (term1 + term2 + term3)
        return Trr

# ------------------------------
# Factory
# ------------------------------

class EquationFactory:
    _REGISTRY: Dict[str, Any] = {
        "airy_stress": AiryStressEquation
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEquation:
        if name not in cls._REGISTRY:
            raise ValueError(f"Equação '{name}' não registrada.")
        return cls._REGISTRY[name](**kwargs)
