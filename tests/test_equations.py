import torch
import pytest
from src.core.equations import AiryStressEquation

# Mock do solver para isolar os testes da equação
class MockSolver:
    def _grad(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

@pytest.fixture
def mock_solver():
    return MockSolver()

@pytest.fixture
def equation():
    return AiryStressEquation()

def test_trr_calculation(equation, mock_solver):
    """
    Testa o cálculo da tensão radial (Trr) com a implementação original.
    A implementação original é d/dr(phi/r).
    Se phi(r) = r^3, então phi/r = r^2.
    A derivada d/dr(r^2) = 2r.
    """
    r = torch.tensor([[2.0]], requires_grad=True)
    phi = r**3
    
    trr_calculated = equation.trr(phi, r, mock_solver)
    
    # Para r=2.0, o resultado esperado é 2 * r = 4.0
    assert torch.isclose(trr_calculated, torch.tensor(4.0))

def test_ttt_calculation(equation, mock_solver):
    """
    Testa o cálculo da tensão tangencial (Ttt) com uma função phi conhecida.
    Se phi(r) = r^3, então d(phi)/dr = 3r^2 e d^2(phi)/dr^2 = 6r.
    Ttt = d^2(phi)/dr^2 = 6r.
    """
    r = torch.tensor([[2.0]], requires_grad=True)
    phi = r**3
    
    ttt_calculated = equation.ttt(phi, r, mock_solver)
    
    # O resultado esperado é 6 * r = 6 * 2.0 = 12.0
    assert torch.isclose(ttt_calculated, torch.tensor(12.0))

def test_residual_calculation_sanity_check(equation, mock_solver):
    """
    Teste de sanidade para o cálculo do resíduo com a lógica original.
    Verifica se a função executa sem erros e produz um resultado não nulo
    para uma entrada que não deveria anular a equação.
    """
    r = torch.tensor([[1.5]], requires_grad=True)
    
    # Simula a relação entre r e r_in como no solver para conectar o grafo
    scale = 1.0 / (2.0 - 1.0)
    shift = 1.0
    r_in = (r - shift) * scale
    
    # A função phi deve depender de r_in, como no solver
    phi = r_in**4
    
    try:
        residual = equation.residual(phi, r, r_in, mock_solver)
        # Apenas verificamos que a execução foi bem-sucedida e o tensor tem o formato certo.
        assert residual.shape == (1, 1)
        # Verificamos que o resultado não é zero, como esperado para phi=r^4
        assert not torch.isclose(residual, torch.tensor(0.0))
    except Exception as e:
        pytest.fail(f"O cálculo do resíduo levantou uma exceção inesperada: {e}")

def test_analytical_solution_known_values():
    """
    Testa a solução analítica com valores de contorno simples.
    Este é um teste mais complexo e pode precisar de valores de referência
    calculados manualmente ou de outra fonte confiável.
    Por enquanto, vamos apenas garantir que a função não retorna erro.
    """
    r = torch.linspace(1.0, 2.0, 10)
    a = 1.0
    b = 2.0
    M = 1.0  # Momento arbitrário

    # Apenas verifica se a execução ocorre sem erros
    try:
        trr_analytical = AiryStressEquation.T_rr_analytical(r, a, b, M)
        assert trr_analytical.shape == r.shape
        assert not torch.isnan(trr_analytical).any()
    except Exception as e:
        pytest.fail(f"A função T_rr_analytical levantou uma exceção inesperada: {e}")
