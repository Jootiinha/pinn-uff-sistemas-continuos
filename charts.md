# Explicação dos Gráficos

Este documento explica os gráficos gerados pelo projeto.

## 1. Gráfico de Convergência do Treinamento (`create_trainning_graph`)

Este gráfico é fundamental para avaliar o processo de treinamento da Rede Neural Informada pela Física (PINN). Ele exibe a evolução das diferentes funções de perda (loss) ao longo das épocas de treinamento.

-   **Eixo X**: Representa as épocas de treinamento.
-   **Eixo Y**: Representa o valor da perda (loss) em escala logarítmica.
-   **Linhas**:
    -   `Loss total`: A soma das perdas da PDE e das condições de contorno. É a métrica principal que o otimizador tenta minimizar.
    -   `PDE loss`: Mede o quanto a saída da rede neural viola a equação diferencial parcial (PDE) subjacente. Um valor baixo indica que a solução encontrada respeita a física do problema.
    -   `BC loss`: Mede o quanto a saída da rede neural viola as condições de contorno (Boundary Conditions) do problema. Um valor baixo indica que a solução respeita as restrições nas fronteiras do domínio.

**Objetivo**: O ideal é que todas as curvas de perda diminuam e convirjam para valores baixos e estáveis, indicando que o modelo aprendeu a satisfazer tanto a equação física quanto as condições de contorno.

## 2. Gráfico da Solução da PINN (`create_phi_graph`)

Este gráfico mostra a solução `phi(r)` que foi aprendida pela PINN.

-   **Eixo X**: Representa a coordenada espacial `r`.
-   **Eixo Y**: Representa o valor da solução `phi(r)` predita pela rede.
-   **Linha**:
    -   `PINN`: A solução para `phi(r)` encontrada pela rede neural.

**Objetivo**: Visualizar a forma da solução encontrada pela PINN для o problema em questão. Frequentemente, este gráfico é comparado com uma solução analítica (se disponível) ou com dados experimentais para validar a precisão do modelo.
