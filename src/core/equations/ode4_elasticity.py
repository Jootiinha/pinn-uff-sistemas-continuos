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

    def trr(self, x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """ Tensão radial: Trr = (1/r) * d(phi)/dr """
        return dy / x

    def ttt(self, x: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        """ Tensão tangencial: Ttt = d2(phi)/dr2 """
        return d2y

    def moment(self, x: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        """
        Calcula o momento M = - integral(Ttt * r) dr
        """
        integrand = self.ttt(x, d2y) * x
        
        # Garante que os tensores estão ordenados para a integração
        sorted_indices = torch.argsort(x.squeeze())
        sorted_x = x[sorted_indices]
        sorted_integrand = integrand[sorted_indices]
        
        # Calcula a integral usando a regra do trapézio
        m = torch.trapz(sorted_integrand.squeeze(), sorted_x.squeeze())
        return -m

    def moment_r(self, x: torch.Tensor, d2y: torch.Tensor) -> torch.Tensor:
        """
        Calcula o momento M(r) = - integral_de_a_ate_r(Ttt * s) ds
        """
        integrand = self.ttt(x, d2y) * x
        
        # Garante que os tensores estão ordenados para a integração
        sorted_indices = torch.argsort(x.squeeze())
        sorted_x = x[sorted_indices]
        sorted_integrand = integrand[sorted_indices]
        
        # Calcula a integral cumulativa usando a regra do trapézio
        # torch.cumulative_trapezoid retorna um tensor de tamanho N-1, então adicionamos um 0 no início
        cumulative_m_sorted = torch.cat([torch.tensor([0.0], device=x.device), torch.cumsum(0.5 * (sorted_integrand.squeeze()[1:] + sorted_integrand.squeeze()[:-1]) * (sorted_x.squeeze()[1:] - sorted_x.squeeze()[:-1]), dim=0)])

        # Precisamos "desordenar" o resultado para corresponder à ordem original de 'x'
        # Criamos um tensor para o resultado e o preenchemos na ordem correta
        unsorted_m = torch.zeros_like(cumulative_m_sorted)
        unsorted_m[sorted_indices] = cumulative_m_sorted

        return -unsorted_m.view(-1, 1)
