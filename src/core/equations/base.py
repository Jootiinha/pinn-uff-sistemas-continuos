import torch


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
