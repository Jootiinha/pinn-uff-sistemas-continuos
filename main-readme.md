
Treinamento de uma Physics-Informed Neural Network (PINN) para resolver 
uma equação diferencial de 4ª ordem:

    d^4(phi)/dr^4 = 0

Domínio: r ∈ [1.0, 2.0]

Condições de contorno:
- Trr(a) = 0 em r = 1.0
- Ttt(b) = 0 em r = 2.0

Configuração do treino:
- Épocas: 2000
- Pontos de colocation (resíduo PDE): 256
- Otimizador: Adam com learning rate 1e-3
- Arquitetura da rede: 4 camadas ocultas, 64 neurônios por camada
- Pesos da loss: PDE = 1.0, BC = 1.0
- Normalização do domínio: desativada

Fluxo do script:
1. Cria a equação via EquationFactory.
2. Define condições de contorno usando StressBC.
3. Configura o solver PINNODE4Solver.
4. Treina a rede, logando perdas PDE e BC a cada N épocas.
5. Avalia a rede no domínio e gera gráficos de convergência e solução.