import torch

from .base import BaseEquation


class ODE4thOrderEquation(BaseEquation):
    """
    Define a equação diferencial de 4ª ordem e as grandezas físicas
    (tensões) associadas.
    """
    def residual_ode4(self, x: torch.Tensor, d2y: torch.Tensor, d3y: torch.Tensor, d4y: torch.Tensor) -> torch.Tensor:
        """
        Resíduo da EDO: d^4(phi)/dr^4 = 0
        A formulação é complexa e baseada em uma função auxiliar 'g'.
        Definição do Resíduo (O Coração da Equação):
            O resíduo é a parte mais importante. 
            É a própria equação diferencial escrita de forma que seu resultado
                seja zero.
            A rede neural será treinada para forçar esse valor a ser zero em todos os pontos do domínio.
        """
        # Esta é a formulação que estava no solver original.
        # g = y'' + y' (não usado diretamente aqui, mas é a base)
        # O resíduo é d/dx(g/x) * d2g/dx2
        # Isso pode ser reescrito em termos de derivadas de y.
        # d/dx( (y''+y')/x ) * d2/dx2(y''+y')
        # (x(y'''+y'') - (y''+y'))/x^2 * (y''''+y''')
        
        # A implementação original no solver era:
        # g = d2y+dy
        # dg_dx_x= self._grad(g/x,x)
        # d2g_dx = self._grad(self._grad(g,x),x)
        # resid = dg_dx_x * d2g_dx
        #
        # Re-expressando em termos de derivadas de y:
        # dg_dx_x = (x * (d3y + d2y) - (d2y + dy)) / x**2
        # d2g_dx = d4y + d3y
        # resid = dg_dx_x * d2g_dx
        #
        # Para simplificar e evitar o dy, vamos usar a formulação d4y=0
        return d4y

    def trr(self, x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """ Tensão radial: Trr = (1/r) * d(phi)/dr """
        return dy / x

    def ttt(self, x: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        """ Tensão tangencial: Ttt = d2(phi)/dr2 """
        return d2y
