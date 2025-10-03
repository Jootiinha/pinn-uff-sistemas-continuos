# pinn-uff-sistemas-continuos
# PINN: Solução de EDO 4ª ordem (var4phi = 0)

Este repositório implementa e treina uma **Physics-Informed Neural Network (PINN)** 
para resolver a equação diferencial de 4ª ordem:

\[\frac{d^4 \varphi}{dr^4} = 0\]

no domínio \( r \in [1.0, 2.0] \), com condições de contorno baseadas nas tensões radiais.

---

## Configuração do Problema

- **Equação:** `var4phi = 0` (quarta derivada de φ)
- **Domínio:** \( r \in [1.0, 2.0] \)
- **Condições de contorno:**
  - \( T_{rr}(a) = 0 \) em \( r = 1.0 \)
  - \( T_{tt}(b) = 0 \) em \( r = 2.0 \)

As condições de contorno são definidas via `StressBC`, associando o operador de tensão ao ponto da fronteira.

---
