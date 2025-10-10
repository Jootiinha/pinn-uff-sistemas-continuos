# Documentação Detalhada do Pacote `solvers`

## 1. Visão Geral: O Poder das PINNs

O pacote `src/core/solvers/` é o coração deste projeto, implementando *solvers* baseados em **Redes Neurais Informadas pela Física (PINNs)**. Uma PINN é uma rede neural treinada para resolver problemas governados por equações diferenciais. A sua principal inovação é a incorporação das próprias equações diretamente na função de perda durante o treinamento.

Isto é feito através de dois componentes de perda principais:
1.  **Perda Informada pela Física**: Penaliza a rede se a sua saída não satisfaz a equação diferencial em um conjunto de pontos de colocação (pontos amostrados no domínio do problema).
2.  **Perda Guiada por Dados**: Penaliza a rede se a sua saída não respeita as condições de contorno ou iniciais conhecidas.

O resultado é uma rede neural que não apenas se ajusta aos dados, mas também aprende a respeitar as leis da física subjacentes, permitindo-lhe encontrar soluções para equações complexas de forma robusta.

## 2. Arquitetura e Padrões de Projeto

Para organizar os solvers de forma eficiente e extensível, o pacote `src/core/solvers/` adota uma arquitetura modular e utiliza o padrão de projeto **Template Method**.

### Estrutura do Pacote

-   **`base.py`**: Define a classe abstrata `PINNSolver`, que implementa o esqueleto do algoritmo de treinamento de uma PINN para EDOs.
-   **`ode2.py`** e **`ode4.py`**: Contêm as implementações concretas (`PINNODE2Solver`, `PINNODE4Solver`) que herdam de `PINNSolver` e preenchem os detalhes específicos para EDOs de 2ª e 4ª ordem, respectivamente.
-   **`algebraic.py`**: Contém o `PINNAlgebraicSolver`, um solver com uma abordagem diferente, projetado para encontrar raízes de equações algébricas.
-   **`__init__.py`**: Consolida as importações, permitindo que todas as classes de solver sejam acessadas diretamente a partir de `src.core.solvers`.

### Padrão de Projeto: Template Method em `PINNSolver`

A classe `PINNSolver` utiliza o padrão **Template Method**. O método `train` na classe base define a estrutura geral e invariável do processo de treinamento de uma PINN:

1.  Iterar por um número de épocas.
2.  Chamar um método abstrato (`_loss_batch`) para calcular a perda.
3.  Realizar a retropropagação (backpropagation).
4.  Atualizar os pesos do modelo.
5.  Registrar o progresso.

As subclasses (`PINNODE2Solver`, `PINNODE4Solver`) são então obrigadas a implementar o método `_loss_batch`, fornecendo a lógica específica para calcular a perda de suas respectivas equações, sem precisar reescrever todo o loop de treinamento. Isso promove a reutilização de código e a consistência entre os solvers.

---

## 3. Componentes Detalhados

### `PINNSolver` (Classe Base em `base.py`)

Esta classe fornece a infraestrutura compartilhada por todos os solvers de EDOs.

#### Funcionalidades Chave

-   **`__init__(...)`**: Inicializa os componentes essenciais:
    -   **Modelo (`MLP1D`)**: Uma rede neural multicamadas (MLP) simples que mapeia uma entrada `x` para uma saída `y(x)`.
    -   **Otimizador (`Adam`)**: O algoritmo usado para minimizar a função de perda.
    -   **Parâmetros de Normalização**: `_scale` e `_shift` são calculados para mapear o domínio do problema (ex: `[a, b]`) para `[0, 1]`. Isso estabiliza o treinamento e acelera a convergência.

-   **`_grad(y, x)`**: O motor da PINN. Este método utiliza a diferenciação automática do PyTorch (`torch.autograd.grad`) para calcular a derivada da saída da rede (`y`) em relação à sua entrada (`x`). Ao aplicar este método recursivamente, podemos obter derivadas de qualquer ordem (`d²y/dx²`, `d³y/dx³`, etc.), que são necessárias para calcular o resíduo da equação diferencial.

-   **`train(...)`**: O "template method". Orquestra o loop de treinamento, chamando `_loss_batch` a cada passo para obter a perda e, em seguida, atualizando o modelo.

-   **`_loss_batch()`**: Método abstrato que deve ser implementado pelas subclasses. É aqui que a "mágica" da PINN acontece, combinando a perda da física com a perda dos dados.

### `PINNODE2Solver` e `PINNODE4Solver` (em `ode2.py` e `ode4.py`)

Estas classes são implementações concretas do `PINNSolver`. Sua principal responsabilidade é implementar o método `_loss_batch`.

#### Implementação de `_loss_batch`

1.  **Amostragem**: Gera um lote de pontos de colocação `x` aleatoriamente dentro do domínio da EDO.
2.  **Forward Pass e Derivadas**:
    -   A rede neural calcula a solução candidata `y = model(x)`.
    -   O método `_grad` é usado para calcular todas as derivadas necessárias para a EDO (até 2ª ordem para `PINNODE2Solver`, até 4ª ordem para `PINNODE4Solver`). A regra da cadeia é aplicada para corrigir as derivadas devido à normalização da entrada.
3.  **Cálculo da Perda da Física (`loss_pde`)**:
    -   As saídas da rede (`y`) e suas derivadas (`dy/dx`, `d²y/dx²`, etc.) são substituídas na equação diferencial para calcular o **resíduo**.
    -   `loss_pde` é a média dos quadrados do resíduo. Minimizar essa perda força a rede a aprender uma função que satisfaz a EDO.
4.  **Cálculo da Perda dos Dados (`loss_bc`)**:
    -   Para cada condição de contorno, a perda é calculada como o erro quadrático entre a previsão da rede e o valor alvo.
    -   **`DirichletBC`**: Força `y(x_b)` a ser igual a `y_b`.
    -   **`NeumannBC`**: Força `y'(x_b)` a ser igual a `g_b`.
    -   **`StressBC` (`PINNODE4Solver`)**: Uma condição mais complexa que força uma função de "stress" (que pode depender de `y` e suas derivadas) a atingir um valor alvo.
5.  **Perda Total**: É a soma ponderada `w_pde * loss_pde + w_bc * loss_bc`, balanceando a importância de satisfazer a física e os dados.

### `PINNAlgebraicSolver` (em `algebraic.py`)

Este solver aborda um problema diferente: encontrar as múltiplas raízes de uma equação algébrica (ex: `ax² + bx + c = 0`).

#### Arquitetura e Estratégia

-   **Modelo (`MLPBranches`)**: Em vez de uma única saída, este modelo tem múltiplos "ramos" de saída. A hipótese é que, com o treinamento correto, cada ramo pode convergir para uma raiz diferente da equação.
-   **Função de Perda Composta**: A função de perda é uma combinação inteligente de três termos:
    1.  **Perda do Resíduo (`loss_res`)**: Garante que cada saída de ramo seja uma raiz válida da equação (ou seja, `f(y_i) ≈ 0`).
    2.  **Perda de Diversidade (`loss_div`)**: Incentiva os ramos a se afastarem uns dos outros. Isso é feito maximizando a variância entre as saídas dos diferentes ramos, evitando que todos convirjam para a mesma raiz.
    3.  **Perda de Intervalo (`loss_range`)**: Uma restrição opcional para manter as soluções dentro de um intervalo esperado.

Essa abordagem transforma a busca de raízes em um problema de otimização, onde a função de perda guia os diferentes ramos do modelo para as múltiplas soluções da equação simultaneamente.
