# PINN para Solução de EDO de 4ª Ordem

Este repositório implementa uma **Rede Neural Informada pela Física (Physics-Informed Neural Network - PINN)** para resolver uma equação diferencial ordinária (EDO) de 4ª ordem. O projeto é desenvolvido em Python utilizando PyTorch e Poetry para gerenciamento de dependências.

---

## O Problema Físico: Elasticidade em Coordenadas Polares

O problema abordado é derivado da teoria da elasticidade para um material sob tensão em coordenadas polares. A equação que governa o comportamento do potencial `phi(r)` é uma EDO de 4ª ordem:

\[ \frac{d^4 \varphi}{dr^4} = 0 \]

Esta equação é resolvida no domínio radial \( r \in [1.0, 2.0] \).

As condições de contorno são baseadas nas tensões físicas do material:
- **Tensão Radial Nula na Borda Interna:** \( T_{rr}(a) = 0 \) em \( r = 1.0 \)
- **Tensão Tangencial Nula na Borda Externa:** \( T_{tt}(b) = 0 \) em \( r = 2.0 \)

A PINN é treinada para encontrar uma função `phi(r)` que satisfaça tanto a EDO quanto as condições de contorno especificadas.

---

## Como Rodar o Projeto

### Pré-requisitos

- **Python:** Versão 3.11 ou superior (até 3.13).
- **Poetry:** Uma ferramenta para gerenciamento de dependências em Python. Se não tiver, instale com `pip install poetry`.

### Passos para Execução

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Jootiinha/pinn-uff-sistemas-continuos.git
    cd pinn-uff-sistemas-continuos
    ```

2.  **Instale as dependências:**
    Use o Poetry para criar um ambiente virtual e instalar todas as bibliotecas necessárias, como PyTorch e Matplotlib.
    ```bash
    poetry install
    ```

3.  **Execute o treinamento:**
    O comando a seguir inicia o script principal (`src/app/main.py`), que treina a rede neural, salva os logs e gera os gráficos da solução. A flag `-B` evita a criação de diretórios `__pycache__`.
    ```bash
    poetry run poe start
    ```
    Alternativamente, você pode ativar o ambiente virtual com `poetry shell` e rodar o script diretamente:
    ```bash
    poetry shell
    python -B src/app/main.py
    ```

### Saídas do Projeto

Após a execução, os seguintes arquivos serão gerados no diretório `docs/`:
- `pde_4th_order_log.csv`: Log da convergência da função de perda.
- `pde_4th_order_metrics.png`: Gráfico da evolução das perdas durante o treinamento.
- `pde_4th_order_phi.png`: Gráfico da solução `phi(r)` encontrada pela PINN.

---

## Estrutura do Projeto

O código está organizado da seguinte forma:

- `src/app/main.py`: Ponto de entrada da aplicação, onde o experimento é configurado e executado.
- `src/core/`: Contém a lógica principal da PINN.
  - `models.py`: Define as arquiteturas das redes neurais.
  - `equations.py`: Implementa as equações diferenciais a serem resolvidas.
  - `solvers.py`: Contém os solvers que treinam as redes.
  - `graphs.py`: Funções para gerar os gráficos de resultado.
- `src/configs/`: Arquivos de configuração para treinamento e condições de contorno.
- `src/experiments/`: Configurações de experimentos específicos.
