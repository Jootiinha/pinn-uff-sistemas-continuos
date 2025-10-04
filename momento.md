# Descrição da Implementação: PINN para Elasticidade

Este documento descreve a arquitetura e o funcionamento da implementação de uma **Rede Neural Informada pela Física (Physics-Informed Neural Network - PINN)** para resolver uma equação diferencial ordinária (EDO) de 4ª ordem, derivada de um problema de elasticidade.

---

## 1. Visão Geral do Problema

O objetivo do projeto é encontrar a solução `phi(r)` para a EDO de 4ª ordem:

\[ \frac{d^4 \varphi}{dr^4} = 0 \]

Esta equação modela o comportamento de um potencial em um material sob tensão em coordenadas polares, no domínio radial \( r \in [a, b] \). A solução deve também satisfazer um conjunto de condições de contorno, que podem incluir valores da função, suas derivadas, ou quantidades físicas como tensões e momentos.

## 2. Arquitetura da Solução

A solução é implementada em Python, utilizando PyTorch como framework de deep learning. A estrutura do projeto pode ser dividida em três camadas principais:

### a. A Rede Neural (O "Coração" da PINN)

-   **Modelo:** A solução `phi(r)` é aproximada por uma rede neural do tipo **Perceptron Multicamadas (MLP)**, definida na classe `MLP1D` (`src/core/models.py`).
-   **Arquitetura:** É uma rede neural densa padrão, que recebe como entrada a coordenada radial `r` e retorna como saída o valor previsto de `phi(r)`. Ela é composta por camadas lineares intercaladas com funções de ativação `Tanh`.
-   **Diferenciação Automática:** A principal vantagem de usar uma rede neural é a capacidade de calcular derivadas de qualquer ordem (`d(phi)/dr`, `d2(phi)/dr2`, etc.) de forma precisa e eficiente, utilizando o mecanismo de **diferenciação automática** do PyTorch (`torch.autograd.grad`).

### b. O Solver (O "Cérebro" da PINN)

-   **Classe Principal:** A lógica de treinamento está encapsulada na classe `PINNODE4Solver` (`src/core/solvers/ode4.py`).
-   **Função de Perda (Loss Function):** O treinamento da rede é guiado pela minimização de uma função de perda composta, que força a rede a aprender a física do problema. A perda total é uma soma ponderada de duas componentes:
    1.  **Perda da EDO (`loss_pde`):** Garante que a solução da rede satisfaça a equação diferencial. O solver calcula o resíduo \( \frac{d^4 \varphi}{dr^4} \) em um conjunto de pontos aleatórios (pontos de colocação) no domínio e busca minimizar o erro quadrático médio desse resíduo.
    2.  **Perda das Condições de Contorno (`loss_bc`):** Garante que a solução respeite as restrições físicas do problema. O solver é flexível e suporta vários tipos de condições:
        -   **`DirichletBC`**: Impõe um valor específico para `phi` em um ponto da fronteira.
        -   **`NeumannBC`**: Impõe um valor específico para a derivada `d(phi)/dr` em um ponto.
        -   **`StressBC`**: Impõe condições sobre as tensões físicas (`Trr` ou `Ttt`), que são funções de `phi` e suas derivadas.
        -   **`MomentBC`**: Uma condição de contorno **integral**. Ela calcula o momento `M = - integral(Ttt * r) dr` sobre todo o domínio e o compara com um valor alvo. A integral é calculada numericamente com a regra do trapézio (`torch.trapz`).

### c. Orquestração e Configuração

-   **Ponto de Entrada:** O script `src/app/main.py` serve como o orquestrador do processo. Ele lê a configuração, instancia o solver, executa o treinamento e salva os resultados.
-   **Configuração Flexível:** Os experimentos são definidos em arquivos de configuração **YAML** (ex: `src/experiments/pde_4th_order/experiment.yaml`). Isso permite modificar facilmente a equação, o domínio, as condições de contorno e os hiperparâmetros de treinamento (taxa de aprendizado, número de épocas, etc.) sem alterar o código principal.
-   **Equações Físicas:** As definições matemáticas da EDO, das tensões (`Trr`, `Ttt`) e do momento (`M`) estão implementadas na classe `ODE4thOrderEquation` (`src/core/equations/ode4_elasticity.py`).

## 3. Fluxo de Execução

1.  O usuário executa `python src/app/main.py` passando o caminho para um arquivo de configuração YAML.
2.  O `main.py` carrega a configuração e instancia o `PINNODE4Solver` com a equação, as condições de contorno e os parâmetros de treino.
3.  O método `train()` do solver é chamado. Em cada época de treinamento:
    -   Um lote de pontos de colocação é gerado.
    -   A rede neural prediz `phi(r)` para esses pontos.
    -   As derivadas até a 4ª ordem são calculadas via diferenciação automática.
    -   A perda da EDO e a perda das condições de contorno são calculadas.
    -   A perda total é retropropagada, e os pesos da rede são atualizados pelo otimizador (ex: Adam).
4.  Após o treinamento, o `main.py` salva o modelo treinado, gera gráficos da convergência da perda e da solução `phi(r)` encontrada, e calcula o valor final do momento `M`.
