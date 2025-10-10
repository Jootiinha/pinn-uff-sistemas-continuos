# Documentação do Projeto PINN

Este projeto implementa uma Rede Neural Informada pela Física (PINN) para resolver equações diferenciais ordinárias (EDOs), com um foco específico em uma EDO de 4ª ordem derivada de um problema de elasticidade.

## Estrutura do Projeto

O código está organizado nos seguintes diretórios e arquivos:

- `src/app/main.py`: O ponto de entrada da aplicação.
- `src/core/`: Contém a lógica principal do solver PINN.
  - `models.py`: Define as arquiteturas das redes neurais (MLPs).
  - `equations.py`: Define as classes que representam as equações a serem resolvidas.
  - `solvers.py`: Implementa os solvers PINN que treinam as redes para resolver as equações.
  - `graphs.py`: Funções para gerar gráficos dos resultados.
- `src/configs/`: Contém os arquivos de configuração.
  - `train_configs.py`: Define as configurações para o treinamento dos modelos.
  - `bc.py`: Define as classes para as condições de contorno.
- `src/experiments/`: Contém as configurações para experimentos específicos.
  - `pde_4th_order/config.py`: Configuração para o problema da EDO de 4ª ordem.

## Detalhamento dos Arquivos

### `src/app/main.py`

Este é o script principal que executa o treinamento da PINN. Ele:
1.  Importa as configurações do experimento (`src/experiments/pde_4th_order/config.py`).
2.  Instancia o solver `PINNODE4Solver` com a equação, a configuração de treino e as condições de contorno.
3.  Inicia o treinamento da rede, salvando um log da convergência em `docs/pde_4th_order_log.csv`.
4.  Gera um gráfico da convergência da função de perda e o salva em `docs/pde_4th_order_metrics.png`.
5.  Usa a rede treinada para prever a solução `phi(r)` no domínio.
6.  Gera um gráfico da solução prevista e o salva em `docs/pde_4th_order_phi.png`.

### `src/core/models.py`

Define as arquiteturas das redes neurais (modelos) usadas no projeto.

-   **`MLP1D`**: Um Perceptron Multicamadas (MLP) padrão, usado para aproximar a solução da EDO. Ele recebe uma coordenada `x` e retorna o valor previsto `y(x)`. A profundidade, largura e função de ativação da rede são configuráveis.
-   **`MLPBranches`**: Um MLP com múltiplas saídas (ramos), projetado para resolver sistemas de equações algébricas.

### `src/core/equations.py`

Este arquivo define as equações que a PINN tentará resolver.

-   **`BaseEquation`**: Uma classe base abstrata que define a interface que todas as equações devem seguir.
-   **`ODE4thOrderEquation`**: Representa a EDO de 4ª ordem `d^4(phi)/dr^4 = 0`. Além de definir o resíduo da EDO, ela também fornece métodos para calcular grandezas físicas (tensões `trr` e `ttt`), que são usadas para definir as condições de contorno.
-   **`EquationFactory`**: Um padrão de projeto *Factory* que permite criar objetos de equação dinamicamente a partir de um nome, desacoplando o código do solver das implementações específicas das equações.

### `src/core/solvers.py`

Contém a lógica central para treinar as PINNs.

-   **`PINNSolver`**: Uma classe base que implementa o fluxo de treinamento geral:
    -   Loop de otimização (épocas).
    -   Cálculo da função de perda (combinando a perda da EDO e a perda das condições de contorno).
    -   Uso de `torch.autograd.grad` para calcular as derivadas necessárias para o resíduo da EDO.
    -   Geração de logs de treinamento.
    -   Método `predict` para usar a rede treinada.
-   **`PINNODE4Solver`**: Uma subclasse que herda de `PINNSolver` e é especializada para a EDO de 4ª ordem. Sua principal responsabilidade é implementar o método `_loss_batch`, que:
    1.  Calcula as derivadas de `phi` até a 4ª ordem (`d_phi`, `d2_phi`, `d3_phi`, `d4_phi`) usando diferenciação automática.
    2.  Calcula o resíduo da EDO.
    3.  Calcula a perda para cada condição de contorno (`StressBC`), que envolve o cálculo das tensões `trr` e `ttt`.
    4.  Combina as perdas da EDO e das condições de contorno em uma única perda total.

### `src/core/graphs.py`

Fornece funções utilitárias para a visualização dos resultados usando `matplotlib`.

-   **`create_trainning_graph`**: Plota a evolução das perdas (total, EDO, BCs) ao longo das épocas de treinamento, permitindo analisar a convergência do modelo.
-   **`create_phi_graph`**: Plota a solução `phi(r)` encontrada pela PINN, mostrando o comportamento da função no domínio do problema.

### `src/configs/`

Este diretório centraliza todas as configurações do projeto.

-   **`bc.py`**: Define `dataclasses` para representar diferentes tipos de condições de contorno (Dirichlet, Neumann e a `StressBC` customizada).
-   **`train_configs.py`**: Define `dataclasses` para agrupar todos os hiperparâmetros de treinamento, como taxa de aprendizado, número de épocas, arquitetura da rede, etc.

### `src/experiments/pde_4th_order/config.py`

Este arquivo configura um experimento específico para resolver a EDO de 4ª ordem. Ele define:
-   O **domínio** do problema (`a`, `b`).
-   A **equação** a ser resolvida, criada via `EquationFactory`.
-   As **condições de contorno** (`StressBC`) a serem impostas nas bordas do domínio.
-   A **configuração de treino** (`TrainConfigODE4`) com todos os hiperparâmetros para este experimento.

Essa estrutura permite que novos experimentos com diferentes EDOs ou configurações sejam facilmente adicionados, simplesmente criando um novo arquivo de configuração no diretório `experiments`.
