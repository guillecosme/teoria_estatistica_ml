# Conceitos detalhados — Tech Challenge Fase 1 (NPS Preditivo)

Referência didática completa de **todos os conceitos** que apareceram no projeto, do zero, em PT-BR, com matemática explicada, analogias do dia a dia e mapeamento para o código real do nosso `repo/`.

> **Como usar.** Não leia linearmente — é referência. Cada seção é autossuficiente. Quando um conceito te encurralar num notebook ou numa aula, abre aqui no Ctrl+F.

> **Convenção sobre acrônimos.** Toda sigla aparece desambiguada na primeira ocorrência da seção em que ela é usada. Em fórmulas, todo símbolo é nomeado em PT-BR logo abaixo da equação.

---

## Índice

**Parte I — Fundamentos**
1. [NPS (Net Promoter Score)](#1-nps-net-promoter-score)
2. [Tipos de variável](#2-tipos-de-variável)
3. [Distribuições estatísticas](#3-distribuições-estatísticas)
4. [Estatísticas descritivas](#4-estatísticas-descritivas)
5. [Histograma, density plot e boxplot](#5-histograma-density-plot-e-boxplot)
6. [Categóricas ordinais vs nominais](#6-categóricas-ordinais-vs-nominais)

**Parte II — Inferência estatística**
7. [Correlação de Pearson](#7-correlação-de-pearson)
8. [Causalidade vs correlação](#8-causalidade-vs-correlação)
9. [População vs amostra](#9-população-vs-amostra)
10. [Lei dos Grandes Números e TCL](#10-lei-dos-grandes-números-e-teorema-central-do-limite-tcl)
11. [Intervalo de confiança e p-valor](#11-intervalo-de-confiança-e-p-valor)
12. [Erros Tipo I e Tipo II, alfa e poder](#12-erros-tipo-i-e-tipo-ii-alfa-e-poder)

**Parte III — Testes de hipótese**
13. [Distribuição de Bernoulli e Binomial](#13-distribuição-de-bernoulli-e-binomial)
14. [Teste t de Welch](#14-teste-t-de-welch)
15. [Intervalo de Wilson para proporções](#15-intervalo-de-wilson-para-proporções)
16. [Teste z de proporções](#16-teste-z-de-proporções)
17. [Service Recovery Paradox (conceito)](#17-service-recovery-paradox-conceito-de-cx)

**Parte IV — Regressão linear**
18. [Regressão linear simples — matemática](#18-regressão-linear-simples--matemática-completa)
19. [Mínimos quadrados (OLS)](#19-mínimos-quadrados-ordinários-ols)
20. [R² e MAE — métricas de regressão](#20-r-mae-mse-rmse--métricas-de-regressão)
21. [Resíduos e premissas](#21-resíduos-e-premissas-da-regressão-linear)
22. [Q-Q plot](#22-q-q-plot)
23. [Termo de interação](#23-regressão-com-termo-de-interação)
24. [Multicolinearidade](#24-multicolinearidade)
25. [Heterocedasticidade](#25-heterocedasticidade-e-homocedasticidade)
26. [Encoding de variáveis categóricas](#26-encoding-de-variáveis-categóricas-one-hot-e-dummies)
27. [Matriz de design](#27-matriz-de-design-design-matrix)

**Parte V — Classificação**
28. [Classificação binária e threshold](#28-classificação-binária-e-threshold)
29. [Matriz de confusão](#29-matriz-de-confusão)
30. [Acurácia, precisão, recall, F1](#30-acurácia-precisão-recall-f1)
31. [Curva ROC e AUC (bônus)](#31-curva-roc-e-auc-bônus)
32. [Por que não logística no nosso projeto](#32-por-que-não-regressão-logística)
33. [Discretização (binning)](#33-discretização-binning)

**Parte VI — Boas práticas de ML**
34. [Train/test split](#34-traintest-split)
35. [Validação cruzada (k-fold)](#35-validação-cruzada-k-fold)
36. [Overfitting e underfitting](#36-overfitting-e-underfitting)
37. [Feature engineering vs feature selection](#37-feature-engineering-vs-feature-selection)
38. [Pipelines do sklearn](#38-pipelines-do-sklearn)
39. [Data leakage](#39-data-leakage)

**Parte VII — Ecossistema Python e reprodutibilidade**
40. [Pandas — operações que usamos](#40-pandas--operações-que-usamos)
41. [NumPy e vetorização](#41-numpy-e-vetorização)
42. [Matplotlib + Seaborn](#42-matplotlib--seaborn)
43. [statsmodels vs scikit-learn](#43-statsmodels-vs-scikit-learn)
44. [Jupyter Notebook por dentro](#44-jupyter-notebook-por-dentro)
45. [Reprodutibilidade — seeds, locks, versionamento](#45-reprodutibilidade--seeds-locks-e-versionamento)
46. [CRISP-DM e Cookiecutter Data Science](#46-crisp-dm-e-cookiecutter-data-science)

---
---

# Parte I — Fundamentos

## 1. NPS (Net Promoter Score)

> **Acrônimo:** **NPS** = *Net Promoter Score* (em PT-BR, "índice de promotores líquido"). Criado por Frederick Reichheld em 2003 num paper na *Harvard Business Review*.

**O que é.** Indicador único de satisfação que reduz toda a relação cliente-empresa a uma pergunta: *"Numa escala de 0 a 10, qual a probabilidade de você recomendar nossa empresa para um amigo?"*. A resposta vira uma das três categorias canônicas:

- **Detrator:** nota 0 a 6 (vai falar mal)
- **Neutro:** nota 7 ou 8 (não defende nem ataca)
- **Promotor:** nota 9 ou 10 (vai recomendar espontaneamente)

**Analogia.** Imagine torcida de futebol. Detrator é o torcedor adversário que xinga seu time no botequim; promotor é o torcedor fanático que veste a camisa todo domingo; neutro é o cara que diz "torço pelo time da minha cidade mas não acompanho". O NPS pergunta, para cada cliente que passou pela sua loja, em que arquibancada ele está.

**Matemática.**

```
NPS = (% de promotores) − (% de detratores)
```

Onde:
- **% de promotores** = (clientes promotores ÷ total de respondentes) × 100
- **% de detratores** = (clientes detratores ÷ total de respondentes) × 100
- **Neutros não entram na fórmula** — somem completamente do cálculo (mas seguem no denominador implícito quando você calcula as proporções)

A escala vai de **−100** (todo mundo é detrator) a **+100** (todo mundo é promotor). Na prática, NPS acima de zero é considerado bom, acima de 30 é excelente, acima de 70 é world-class.

**No nosso projeto.** A coluna `nps_score` é o resultado da pergunta acima. No notebook 03 (`features.py`), criamos a coluna `categoria_nps` aplicando os cortes canônicos:

```python
# repo/src/nps/features.py
CORTE_DETRATOR_NEUTRO = 7
CORTE_NEUTRO_PROMOTOR = 9

resultado["categoria_nps"] = pd.cut(
    resultado["nps_score"],
    bins=[-float("inf"), CORTE_DETRATOR_NEUTRO, CORTE_NEUTRO_PROMOTOR, float("inf")],
    labels=["detrator", "neutro", "promotor"],
    right=False,  # intervalo fechado a esquerda: nota 7.0 vira neutro, 6.9 ainda e detrator
    ordered=True,
)
```

**Cuidados.**
- Por que neutros não contam? Reichheld argumenta que neutros são instáveis — pequena coisa os converte para qualquer lado, então não dá para confiar na nota deles. Crítica acadêmica (Keiningham 2007) discorda dessa decisão.
- A pergunta usa a palavra "amigo", não "alguém". Isso é deliberado — força o respondente a pensar em alguém de quem ele se importa.

---

## 2. Tipos de variável

**O que é.** Toda coluna de um dataset tem um "tipo" estatístico que determina o que você pode fazer com ela.

- **Categórica nominal**: rótulos sem ordem natural. Ex.: `customer_region` (Sul, Sudeste, Nordeste — não dá para dizer que Sul > Sudeste).
- **Categórica ordinal**: rótulos com ordem. Ex.: `categoria_nps` (detrator < neutro < promotor).
- **Numérica discreta**: valores inteiros contáveis. Ex.: `customer_service_contacts` (0, 1, 2, 3 contatos — não existe 1,5 contato).
- **Numérica contínua**: valores em escala real, podendo ter casas decimais. Ex.: `order_value` (R$ 127,53).

**Analogia.** Pense num formulário do RH. "Sexo" é nominal (M/F). "Cargo" é ordinal (júnior < pleno < sênior). "Idade" é discreta (você diz 35 anos, não 35,7). "Salário" é contínuo (R$ 12.385,42).

**No nosso projeto.** O tipo da coluna determina **que gráfico fazer**:

| Tipo | Gráfico apropriado | Estatística apropriada |
|---|---|---|
| Nominal | Barras, pizza | Moda, frequência |
| Ordinal | Barras (com ordem) | Mediana, moda |
| Discreta | Barras ou histograma | Média, mediana |
| Contínua | Histograma, density, boxplot | Média, mediana, desvio-padrão |

**Snippet Python.**

```python
# Como pandas guarda cada tipo
dados["customer_region"] = dados["customer_region"].astype("category")  # nominal
dados["categoria_nps"] = pd.Categorical(
    dados["categoria_nps"],
    categories=["detrator", "neutro", "promotor"],
    ordered=True,  # ordinal!
)
```

**Cuidados.**
- Ordinal vira numérica? Em muitos contextos, sim — promover "detrator/neutro/promotor" para 0/1/2 funciona. Em regressão linear isso seria assumir que a "distância" entre detrator e neutro é igual à distância entre neutro e promotor, o que nem sempre é verdade.
- Por que isso importa em código? Se você tem `customer_region` como string, pandas guarda 2.500 cópias da palavra "Sudeste". Como `category`, ele guarda só uma vez e usa códigos inteiros internamente — economia de memória.

---

## 3. Distribuições estatísticas

**O que é.** Distribuição é a forma como os valores de uma variável se espalham pela escala. Você plota um histograma e a "silhueta" que aparece é a distribuição empírica.

**Distribuições teóricas mais comuns:**

### 3.1. Distribuição Normal (Gauss)

A famosa "curva sino". Simétrica em torno da média. A maioria dos valores fica perto do centro, e poucos valores ficam nas pontas.

**Matemática.**

```
f(x) = (1 / (σ × √(2π))) × e^(−(x−μ)² / (2σ²))
```

Onde:
- **μ** (mu) = média da distribuição (centro do sino)
- **σ** (sigma) = desvio-padrão (largura do sino)
- **e** = constante de Euler (~2,71828)
- **π** (pi) = constante (~3,14159)

Você não precisa decorar essa fórmula. O importante é entender: dois números (μ e σ) descrevem o sino inteiro.

**Regra empírica:**
- 68% dos dados ficam dentro de **μ ± 1σ**
- 95% dos dados ficam dentro de **μ ± 2σ**
- 99,7% dos dados ficam dentro de **μ ± 3σ**

**Por que aparece tanto.** O Teorema Central do Limite (TCL — ver seção 10) garante que **médias de amostras** tendem a seguir distribuição normal, mesmo quando a variável original não é normal. Por isso testes estatísticos clássicos (t, z) assumem normalidade nas médias.

### 3.2. Distribuição Assimétrica

Quando a curva pende mais para um lado. Pode ser:
- **Assimétrica à direita (positiva):** cauda longa para a direita, maioria dos valores à esquerda. Ex.: salários no Brasil (poucos ganham muito, maioria ganha pouco).
- **Assimétrica à esquerda (negativa):** cauda longa para a esquerda, maioria à direita. Mais raro.

**No nosso projeto.** A coluna `order_value` tinha assimetria à direita (poucos pedidos muito altos). Aplicamos `log1p` (logaritmo de `1+x`) no notebook 03 para reduzir essa assimetria e estabilizar a variância.

**Snippet.**

```python
import numpy as np
dados["log_order_value"] = np.log1p(dados["order_value"])
# log1p e' melhor que log porque aceita zero (log(0) e' indefinido, log1p(0) = 0)
```

### 3.3. Distribuição Bimodal

Duas "cordas" no sino, em vez de uma. Sinaliza que existem duas subpopulações misturadas. No nosso `nps_score`, a distribuição é bem assimétrica à esquerda (84% detratores), com pico em 0 e cauda subindo até 10.

**Cuidados.**
- Antes de aplicar qualquer teste estatístico que pressupõe normalidade (t-test, regressão), olhe a distribuição num histograma.
- Transformações comuns para corrigir assimetria: `log`, `log1p`, raiz quadrada, Box-Cox.

---



---

*Caderno em construção. Demais seções do índice serão preenchidas conforme o estudo avança.*
