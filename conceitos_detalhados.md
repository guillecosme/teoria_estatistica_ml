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

## 4. Estatísticas descritivas

**O que é.** Números que resumem uma coluna inteira em um valor só.

### 4.1. Medidas de tendência central

- **Média (aritmética):** soma de todos os valores ÷ quantidade. Sensível a outliers.
- **Mediana:** valor do meio quando os dados estão ordenados. Robusta a outliers.
- **Moda:** valor mais frequente. Útil para variáveis categóricas e discretas.

**Analogia.** Você quer saber "quanto o brasileiro ganha".
- Média: pega salário de todo mundo, soma e divide. Resultado puxado para cima por 1% de bilionários. Resposta: "R$ 4.500".
- Mediana: você enfileira todos do mais pobre ao mais rico e pega o do meio. Resposta: "R$ 2.800".
- A diferença entre média e mediana é o **viés de outlier**.

**Matemática.**

```
Média:    x̄ = (Σ xᵢ) / n
Mediana:  valor de posicao (n+1)/2 nos dados ordenados
Moda:     argmax(frequencia(xᵢ))
```

Onde:
- **x̄** (x-barra) = símbolo padrão para média de uma amostra
- **xᵢ** = i-ésimo valor da amostra
- **Σ** (sigma maiúsculo) = somatório (some todos os valores)
- **n** = tamanho da amostra

### 4.2. Medidas de dispersão

- **Variância** (σ²): média dos quadrados das distâncias à média.
- **Desvio-padrão** (σ): raiz da variância. Tem a mesma unidade da variável original (por isso é mais legível que variância).
- **Amplitude:** valor máximo − valor mínimo.
- **Intervalo interquartil (IQR):** Q3 − Q1, onde Q1 é o 25º percentil e Q3 é o 75º.

**Matemática (variância amostral).**

```
σ² = (Σ (xᵢ − x̄)²) / (n − 1)
```

Onde:
- **σ²** (sigma ao quadrado) = variância
- **xᵢ − x̄** = distância de cada valor à média
- **(...)²** = elevamos ao quadrado para que distâncias positivas e negativas não se anulem
- **n − 1** = denominador (em vez de n) é a "correção de Bessel" para estimativa não-viesada da variância populacional a partir de uma amostra

**Analogia.** Pense em peso de duas turmas. Turma A: todos pesam 70 kg. Turma B: metade pesa 50, metade pesa 90. Ambas têm média 70, mas a turma B tem variância muito maior. Desvio-padrão te diz "quão dispersos os dados estão da média".

### 4.3. Percentis

O percentil P é o valor abaixo do qual P% dos dados ficam. Mediana é o percentil 50. Quartis são percentis 25, 50 e 75.

**No nosso projeto.** Em todos os histogramas de variáveis contínuas adicionamos `axvline(media, ...)` (regra explícita do projeto) para ancorar o leitor na média. No notebook 02, descrevemos cada coluna com `dados.describe()`, que retorna média, desvio-padrão, mín, Q1, mediana, Q3 e máx.

**Snippet.**

```python
dados["nps_score"].describe()
# count    2500.000000
# mean        4.380000  <- media
# std         3.123456  <- desvio-padrao
# min         0.000000
# 25%         1.800000  <- Q1
# 50%         4.500000  <- mediana
# 75%         6.900000  <- Q3
# max        10.000000
```

**Cuidados.**
- Use mediana quando a distribuição é assimétrica.
- Variância tem unidade ao quadrado (R$², kg²) — sempre converta para desvio-padrão para comunicar.
- O `.describe()` do pandas usa Q1=25% e Q3=75% por padrão.

---

## 5. Histograma, density plot e boxplot

Três gráficos diferentes para mostrar a mesma coisa: como uma variável se distribui.

### 5.1. Histograma

**O que é.** Você divide a faixa de valores em "caixas" (bins) e conta quantas observações caíram em cada caixa. A altura da barra é a contagem.

**Quando usar.** Sempre que quiser ver a forma da distribuição, identificar pico, assimetria, multi-modalidade.

**Snippet.**

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(dados["nps_score"], bins=22, edgecolor="white")
ax.axvline(dados["nps_score"].mean(), color="black", linestyle=":",
           label=f"média = {dados['nps_score'].mean():.2f}")
ax.legend()
```

### 5.2. Density plot (KDE)

**Acrônimo:** **KDE** = *Kernel Density Estimation* (estimativa de densidade por kernel).

**O que é.** Versão "suavizada" do histograma. Em vez de barras, uma curva contínua. Útil para comparar distribuições de várias variáveis no mesmo gráfico.

**Snippet.**

```python
sns.kdeplot(data=dados, x="nps_score", hue="customer_region")
```

### 5.3. Boxplot

**O que é.** Caixa com 5 informações: mínimo, Q1, mediana, Q3, máximo. Pontos isolados além dos "bigodes" são outliers.

**Componentes:**
- **Caixa:** vai de Q1 até Q3 (50% centrais dos dados).
- **Linha dentro da caixa:** mediana.
- **Bigodes (whiskers):** estendem-se até 1,5 × IQR de cada lado.
- **Pontos fora dos bigodes:** outliers.

**Quando usar.** Comparar distribuições entre grupos. Ex.: NPS por categoria de cliente.

**Snippet.**

```python
sns.boxplot(data=dados, x="categoria_nps", y="customer_service_contacts")
```

**Cuidados.**
- Boxplot esconde o número de observações em cada grupo. Sempre confira `value_counts()` antes.
- Histograma com poucos bins pode esconder estrutura; com muitos bins, vira ruído. A regra prática é começar com 20-30 bins.

---

## 6. Categóricas ordinais vs nominais

**O que é.** Já vimos na seção 2, mas vale aprofundar porque define como você passa essas variáveis para um modelo.

**Ordinal** tem ordem natural. Você pode dizer "X > Y" sem ofender ninguém: detrator < neutro < promotor.

**Nominal** não tem ordem. Sul, Sudeste e Nordeste são só rótulos.

**No nosso projeto.** Em `features.py`, definimos `categoria_nps` como ordinal e `customer_region` como nominal:

```python
# Ordinal
dados["categoria_nps"] = pd.Categorical(
    dados["categoria_nps"],
    categories=["detrator", "neutro", "promotor"],
    ordered=True,
)

# Nominal
dados["customer_region"] = dados["customer_region"].astype("category")  # sem ordered=True
```

**Por que isso importa.**
- Em ordinal, comparações `<` e `>` funcionam: `dados["categoria_nps"] >= "neutro"`.
- Em regressão, ordinais podem ser tratados como numéricas (codificadas 0, 1, 2). Nominais precisam de **one-hot encoding** (ver seção 26).
- Pandas `groupby` em ordinal preserva a ordem; em nominal usa ordem alfabética.

---

---

# Parte II — Inferência estatística

## 7. Correlação de Pearson

**O que é.** Número entre −1 e +1 que mede a **força e direção da associação linear** entre duas variáveis numéricas.

- **+1:** quando uma sobe, a outra sobe na mesma proporção (relação linear perfeita positiva).
- **0:** não há associação linear (mas pode haver não-linear!).
- **−1:** quando uma sobe, a outra desce na mesma proporção (relação linear perfeita negativa).

**Analogia.** Você anota seu peso e seu IMC todo mês. Os dois sobem juntos — correlação positiva alta. Você anota seu peso e o número da casa onde mora — correlação ~0, sem relação. Você anota dia da semana e quantos quilômetros pedalou no fim de semana: a tendência é Sex < Sáb > Dom — uma relação que existe, mas **não é linear**, então Pearson dá quase zero apesar de haver padrão.

**Matemática.**

```
r = Σ((xᵢ − x̄)(yᵢ − ȳ)) / √(Σ(xᵢ − x̄)² × Σ(yᵢ − ȳ)²)
```

Onde:
- **r** = coeficiente de correlação de Pearson
- **xᵢ, yᵢ** = i-ésimos valores das duas variáveis
- **x̄, ȳ** = médias das duas variáveis
- **Numerador:** covariância (mede o quanto x e y "andam juntos")
- **Denominador:** produto dos desvios-padrão (normaliza para a escala 0-1)

A **covariância** sozinha não é interpretável (depende da unidade das variáveis). Pearson normaliza tudo numa escala universal.

**No nosso projeto.** No notebook 04 (seção 4.2), calculamos correlação de Pearson de cada variável numérica com `nps_score`. O atraso na entrega liderou com r ≈ −0,597. O tempo total de entrega deu r ≈ +0,001 — Pearson não detectou nada porque a relação (se existir) é não-linear, motivando o teste de hipótese H1 com Welch.

**Snippet.**

```python
correlacoes = dados.select_dtypes(include="number").corrwith(dados["nps_score"])
correlacoes.sort_values()
```

**Cuidados.**
- Pearson **só captura relação linear**. Se a relação for em U, exponencial ou ter ponto de ruptura, Pearson dá ~0 e te engana.
- Pearson é sensível a outliers. Um único valor extremo pode mover a correlação muito.
- Alternativa: correlação de **Spearman** (correlação por postos), que captura relações monotônicas mesmo não-lineares.

---

## 8. Causalidade vs correlação

**O que é.** Correlação é associação estatística. Causalidade é "X faz Y acontecer".

**Analogia.** Estatística clássica adora o exemplo: vendas de sorvete e afogamentos sobem juntos no verão. Pearson dá correlação positiva alta. Mas sorvete não causa afogamento — a causa comum é o calor, que aumenta as duas coisas independentemente. Isso se chama **variável de confusão** (confounder).

**No nosso projeto.** Mesmo achando correlação forte entre `delivery_delay_days` e `nps_score`, isso não prova que atraso causa baixo NPS. Em tese, poderia haver uma variável escondida (ex.: clima ruim) afetando os dois. A EDA faz seu melhor — testa hipóteses, controla para variáveis de confusão (ex.: H7 testou se região é só proxy de operacional) — mas conclusão definitiva de causalidade exige experimento controlado (A/B test), que não temos.

**Como sair do impasse.**
- **A/B test:** sortear aleatoriamente quem entra na operação melhorada e comparar. Aleatoriedade quebra confundidores.
- **Variáveis de controle:** rodar regressão multivariada incluindo possíveis confundidores e ver se o coeficiente da variável de interesse sobrevive.
- **Análise de mediação:** testar se o efeito de X em Y passa por uma variável intermediária M.

**Cuidados.**
- "Correlation does not imply causation" é o mantra que toda apresentação para gestores precisa carregar.
- Mas... falta de correlação **também** não prova falta de causalidade (relação pode ser não-linear).

---

## 9. População vs amostra

**O que é.** Em estatística, todo dataset é uma **amostra** — um subconjunto — de uma **população** que você quer entender.

- **População:** todos os pedidos que esse e-commerce já fez ou pode fazer.
- **Amostra:** os 2.500 pedidos no nosso CSV.

**Analogia.** Pesquisa eleitoral. Você não pergunta para 200 milhões de brasileiros — pergunta para 2.000. Esses 2.000 são a amostra; o eleitorado todo é a população. Dá para tirar conclusões sobre a população **se** a amostra for representativa.

**No nosso projeto.** Os 2.500 pedidos são amostra de uma operação maior. Por isso usamos:
- **Estatística descritiva** para resumir a amostra (média, desvio).
- **Estatística inferencial** para extrapolar para a população (IC, p-valor, teste de hipótese).

**Por que `n − 1` na fórmula da variância amostral.** Quando você calcula variância usando a média da amostra (que já foi estimada dos próprios dados), você "perde 1 grau de liberdade". Dividir por `n − 1` em vez de `n` corrige esse viés.

**Cuidados.**
- Amostra precisa ser **aleatória e representativa**. Se você só tem dados dos clientes que reclamaram, não dá para generalizar para "todos os clientes".
- Tamanho importa: amostra de 30 traz muito mais incerteza que amostra de 2.500.

---

## 10. Lei dos Grandes Números e Teorema Central do Limite (TCL)

> **Acrônimo:** **TCL** = Teorema Central do Limite (em inglês CLT — *Central Limit Theorem*).

### Lei dos Grandes Números

**O que é.** Conforme o tamanho da amostra cresce, a média amostral converge para a média populacional.

**Analogia.** Você joga uma moeda 10 vezes e tira 7 caras. Estranho? Talvez. Joga 10.000 vezes e tira 5.013 caras. Bem mais perto de 50%. Quanto mais lances, mais a frequência observada se aproxima da probabilidade real.

### Teorema Central do Limite

**O que é.** Se você tira muitas amostras de qualquer distribuição (mesmo bizarra) e calcula a **média de cada amostra**, a distribuição dessas médias é aproximadamente **normal**, com:
- Média igual à média populacional.
- Desvio-padrão = σ_população / √n.

**Por que isso é mágico.** Não importa se a variável original é normal, assimétrica ou multimodal — médias de amostras convergem para a normal. É isso que justifica usar testes baseados em distribuição normal (t, z) para qualquer dataset suficientemente grande.

**No nosso projeto.** Usamos teste t de Welch (que pressupõe normalidade na média) sem nos preocupar muito com a normalidade do `nps_score` original (que é bem assimétrico) **porque** nossas amostras têm centenas de observações. TCL garante que a média amostral está perto da normal.

**Regra prática.** Para a maioria das distribuições, n > 30 já é suficiente para o TCL valer "aproximadamente". Para distribuições muito bizarras, n > 100.

**Cuidados.**
- TCL fala da **distribuição da média**, não da distribuição original. Histograma do `nps_score` pode ser estranho — não importa para o teste t.
- Não vale para amostras muito pequenas (n < 20).

---

## 11. Intervalo de confiança e p-valor

> **Acrônimo:** **IC** = intervalo de confiança.

### Intervalo de confiança

**O que é.** Faixa numérica que provavelmente contém o "valor verdadeiro" do parâmetro populacional.

**Analogia.** Pesquisa eleitoral diz "candidato A tem 35% das intenções, com margem de erro de 3 pontos para cima ou para baixo, com 95% de confiança". O IC é exatamente isso: estimativa pontual (35%) ± margem (3pp).

**Matemática (IC para média, com TCL).**

```
IC95% = x̄ ± 1,96 × (s / √n)
```

Onde:
- **x̄** = média da amostra
- **s** = desvio-padrão da amostra
- **n** = tamanho da amostra
- **1,96** = quantil da distribuição normal padrão para 95% (2 caudas; para 99% seria 2,58)

**Interpretação correta:** "Se eu repetisse esse experimento 100 vezes, em 95 delas o IC calculado conteria a média populacional verdadeira." É uma propriedade do **procedimento**, não do intervalo específico.

**Interpretação errada (mas comum):** "Tem 95% de chance da média estar nesse intervalo." Tecnicamente errado em estatística frequentista, mas todo mundo fala assim.

### p-valor

**O que é.** A probabilidade de você observar uma diferença tão grande (ou maior) que a observada, **assumindo que a hipótese nula é verdadeira**.

**Analogia.** Você joga uma moeda 10 vezes e tira 9 caras. p-valor é "qual a chance de tirar 9+ caras numa moeda honesta?". Se a chance for muito pequena (digamos, 1%), você desconfia que a moeda é viciada.

- **p < 0,05:** evidência fraca contra a hipótese nula. Convencionalmente "rejeitamos" H0.
- **p < 0,01:** evidência forte.
- **p < 0,001:** evidência muito forte.
- **p > 0,05:** não há evidência suficiente para rejeitar H0.

**No nosso projeto.**
- H1 (ponto de ruptura no tempo de entrega): teste t deu p = 0,77 → não rejeitamos H0 → hipótese rejeitada.
- H5 (NPS prediz recompra): teste z deu p < 0,001 → rejeitamos H0 → hipótese confirmada com evidência muito forte.

**Cuidados.**
- p-valor **não é** "a probabilidade de a hipótese ser verdadeira". É a probabilidade dos dados observados, dada a hipótese.
- p-valor é sensível ao tamanho da amostra. Com n = 1 milhão, qualquer diferença minúscula vira "estatisticamente significante" mesmo sem importância prática.
- O limite 0,05 é convenção, não lei da natureza. Áreas críticas (medicina, física) usam 0,001 ou menos.

---

## 12. Erros Tipo I e Tipo II, alfa e poder

**O que é.** Em todo teste de hipótese, você pode errar de duas formas:

- **Erro Tipo I (falso positivo):** rejeitar H0 quando H0 é verdadeira. Você "vê" um efeito que não existe.
- **Erro Tipo II (falso negativo):** não rejeitar H0 quando H0 é falsa. Você não detecta um efeito que existe.

**Analogia médica.** Teste de doença:
- H0: "paciente é saudável".
- Erro Tipo I: alarme falso — diz que tá doente, mas tá saudável.
- Erro Tipo II: doença passou despercebida — diz que tá saudável, mas tá doente.

**Acrônimos:**
- **α** (alfa) = probabilidade de cometer Erro Tipo I. Você define isso antes do teste (geralmente 0,05).
- **β** (beta) = probabilidade de cometer Erro Tipo II.
- **Poder estatístico** = 1 − β. Probabilidade de detectar um efeito real quando ele existe.

**Conexão direta com matriz de confusão.** Erro Tipo I é o **falso positivo (FP)** da nossa classificação. Erro Tipo II é o **falso negativo (FN)**. Mesma ideia, mesma matemática.

**Trade-off.** Reduzir α (ser mais conservador) aumenta β. Para reduzir os dois ao mesmo tempo, precisa **aumentar n** (tamanho da amostra).

**No nosso projeto.** Usamos α = 0,05 implicitamente em todos os testes. Não calculamos poder explicitamente, mas com n = 2.500 nosso poder é alto para efeitos de tamanho médio.

**Cuidados.**
- Antes de fazer um teste, decida α. Mudar α depois de ver o p-valor é trapaça (p-hacking).
- Se você faz muitos testes, a chance de um deles dar falso positivo cresce. Isso se chama **problema de comparações múltiplas** (correção de Bonferroni resolve).

---

---

# Parte III — Testes de hipótese

## 13. Distribuição de Bernoulli e Binomial

### Bernoulli

**O que é.** Distribuição da variável binária mais simples possível: 0 ou 1, sucesso ou fracasso, com probabilidade p de sucesso.

**Exemplos:** uma moeda (50% cara), um cliente (recompra ou não), um ticket (resolvido ou não).

**Matemática.**

```
P(X = 1) = p
P(X = 0) = 1 − p
Média:     E[X] = p
Variância: Var(X) = p(1 − p)
```

Onde:
- **X** = a variável aleatória (0 ou 1)
- **p** = probabilidade de sucesso
- **E[X]** = valor esperado (média teórica)

### Binomial

**O que é.** Soma de n Bernoullis independentes. Conta quantos sucessos em n tentativas.

**Exemplo.** Você joga uma moeda 10 vezes. Quantas caras saíram? Esse número segue uma distribuição binomial com n = 10 e p = 0,5.

**Matemática.**

```
P(X = k) = C(n, k) × p^k × (1 − p)^(n−k)
```

Onde:
- **k** = número de sucessos observados
- **n** = número de tentativas
- **p** = probabilidade de sucesso por tentativa
- **C(n, k)** = "n escolhe k", o número de formas de escolher k itens em n (coeficiente binomial)
- **Média:** E[X] = n × p
- **Variância:** Var(X) = n × p × (1 − p)

**No nosso projeto.** Toda variável binária (`eh_detrator`, `repeat_purchase_30d`) segue Bernoulli. O **número total** de detratores na base segue Binomial. Isso fundamenta os testes de proporção (seções 15 e 16).

**Cuidados.**
- Bernoulli pressupõe independência. Se um cliente influencia outro (efeito de rede), Bernoulli não se aplica direto.

---

## 14. Teste t de Welch

**O que é.** Teste estatístico para comparar **médias de dois grupos** quando você não sabe se as variâncias são iguais.

**Analogia.** Você quer saber se mulheres dão NPS médio diferente de homens. Mede média de cada grupo, calcula a diferença e pergunta: "essa diferença é grande o suficiente para não ser só sorte da amostra?".

**Hipóteses do teste:**
- **H0:** as duas médias são iguais (μ₁ = μ₂).
- **H1:** as duas médias são diferentes (μ₁ ≠ μ₂) — teste bilateral.

**Matemática.**

```
t = (x̄₁ − x̄₂) / √((s₁²/n₁) + (s₂²/n₂))
```

Onde:
- **x̄₁, x̄₂** = médias dos dois grupos
- **s₁², s₂²** = variâncias dos dois grupos
- **n₁, n₂** = tamanhos dos grupos

A estatística `t` segue uma distribuição t de Student com graus de liberdade aproximados pela fórmula de Welch-Satterthwaite (não precisa decorar).

**Por que "Welch" e não "t-student padrão".** O t-student padrão pressupõe variâncias iguais. Welch relaxa essa premissa. Em 99% dos casos da vida real, use Welch — é mais robusto.

**No nosso projeto.** Usamos Welch em H1 para comparar `nps_score` médio entre clientes com entrega rápida vs muito lenta:

```python
# repo/notebooks/04_eda.ipynb
from scipy import stats
grupo_rapido = dados.loc[dados["faixa_entrega"] == "rapida", "nps_score"]
grupo_lento = dados.loc[dados["faixa_entrega"] == "muito_lenta", "nps_score"]
estatistica, p_valor = stats.ttest_ind(grupo_rapido, grupo_lento, equal_var=False)
# equal_var=False ativa o teste de Welch
```

Resultado: p ≈ 0,77 → não rejeitamos H0 → as duas faixas têm NPS médio iguais → hipótese H1 rejeitada.

**Cuidados.**
- Pressupõe TCL (médias dos grupos aproximadamente normais). Para n grande, OK.
- Sensível a outliers (porque média é sensível). Se houver outliers extremos, considere Mann-Whitney (versão não-paramétrica).
- Teste bilateral (`!=`) é o default. Para teste unilateral (`>` ou `<`), divida o p-valor por 2.

---

## 15. Intervalo de Wilson para proporções

**O que é.** Intervalo de confiança específico para proporções (como porcentagem de detratores).

**Por que não usar IC normal.** Para proporções, especialmente quando p é próximo de 0 ou 1, ou quando n é pequeno, o IC clássico baseado em distribuição normal dá resultados ruins (pode ir abaixo de 0 ou acima de 1, o que é absurdo). Wilson corrige isso.

**Matemática (simplificada).** Não precisa decorar — `statsmodels` calcula. O importante é entender que Wilson:
- Sempre fica dentro de [0, 1].
- É mais preciso que IC normal para proporções extremas.
- Continua simétrico em torno de uma versão "ajustada" da proporção observada.

**No nosso projeto.** Usamos Wilson em várias hipóteses sobre proporções:

```python
from statsmodels.stats.proportion import proportion_confint

# Proporcao de detratores entre clientes silenciosos
n_silenciosos = (dados["customer_service_contacts"] == 0).sum()
n_silenciosos_detratores = (
    (dados["customer_service_contacts"] == 0) & (dados["categoria_nps"] == "detrator")
).sum()

low, high = proportion_confint(
    count=n_silenciosos_detratores,
    nobs=n_silenciosos,
    alpha=0.05,
    method="wilson",  # <- escolhe o metodo de Wilson
)
```

**Cuidados.**
- Para proporções no meio (0,3 a 0,7) com n grande, Wilson e IC normal dão resultados quase idênticos. Wilson brilha nas pontas.

---

## 16. Teste z de proporções

**O que é.** Teste para comparar **duas proporções** (ex.: % de recompra entre detratores vs % entre promotores).

**Hipóteses:**
- **H0:** as duas proporções são iguais (p₁ = p₂).
- **H1:** são diferentes.

**Matemática (simplificada).**

```
z = (p̂₁ − p̂₂) / √(p̂(1 − p̂) × (1/n₁ + 1/n₂))
```

Onde:
- **p̂₁, p̂₂** = proporções observadas (chapéu = "estimativa")
- **p̂** = proporção combinada das duas amostras (assumindo H0 verdadeira)
- **n₁, n₂** = tamanhos dos grupos

A estatística `z` segue distribuição normal padrão (média 0, desvio 1).

**No nosso projeto.** Usamos em H5 para comparar recompra entre detratores e promotores:

```python
from statsmodels.stats.proportion import proportions_ztest

contagens = [
    (dados["categoria_nps"] == "promotor") & (dados["repeat_purchase_30d"] == 1),
    (dados["categoria_nps"] == "detrator") & (dados["repeat_purchase_30d"] == 1),
]
totais = [
    (dados["categoria_nps"] == "promotor").sum(),
    (dados["categoria_nps"] == "detrator").sum(),
]
estatistica, p_valor = proportions_ztest(
    count=[c.sum() for c in contagens],
    nobs=totais,
)
```

Resultado: p < 0,001 → diferença gigante (0% vs 100%) → H5 confirmada.

**Cuidados.**
- Pressupõe que as duas amostras são independentes.
- Para amostras pareadas (ex.: cliente antes/depois), use teste de McNemar.

---

## 17. Service Recovery Paradox (conceito de CX)

> **Acrônimo:** **CX** = *Customer Experience* (experiência do cliente).

**O que é.** Conceito de marketing/CX que diz: quando uma empresa **falha** com o cliente mas **recupera bem** (atende rápido, resolve, compensa), o cliente pode ficar **mais satisfeito** do que se nada tivesse dado errado.

**Analogia.** Um restaurante esquece seu pedido. O gerente vem se desculpar, te dá um drink grátis e o prato fica pronto em 5 min. Você sai falando bem do lugar — talvez até melhor do que se a noite tivesse sido normal.

**Por que importa.** Sugere que investir em SAC eficiente (reduzir tempo de resolução, treinar atendentes) compensa: você não evita falhas, mas converte a falha em oportunidade.

**No nosso projeto.** Hipótese H3 testou esse conceito em duas versões:
- **Versão forte:** "SAC rápido leva o detrator de volta ao patamar de quem nunca teve problema." → Rejeitada (p < 0,001, diferença +2,27 pontos de NPS entre quem não reclamou e quem reclamou+resolveu rápido).
- **Versão atenuada:** "SAC rápido amortece o dano em mais de 1 ponto de NPS." → Confirmada (diferença de +1,27 pontos entre SAC rápido e SAC lento entre quem reclamou).

**Cuidados.**
- O paradoxo não vale para todas as falhas. Falhas críticas (acidente grave, dado vazado) raramente são compensadas por bom SAC.
- Existe **viés de sobrevivência**: você só ouve de clientes que ficaram. Quem foi embora silenciosamente não fala.

---

---

# Parte IV — Regressão linear

## 18. Regressão linear simples — matemática completa

**O que é.** Modelo que ajusta uma reta aos dados para descrever como uma variável (Y) varia em função de outra (X).

**Equação.**

```
y = β₀ + β₁ × x + ε
```

Onde:
- **y** = variável dependente (target, o que você quer prever)
- **x** = variável independente (preditor, feature)
- **β₀** (beta zero) = intercepto. Valor de y quando x = 0. Visualmente, onde a reta cruza o eixo y.
- **β₁** (beta um) = coeficiente angular. Inclinação da reta. Diz quanto y muda quando x aumenta em 1 unidade.
- **ε** (epsilon) = erro aleatório. A diferença entre o valor real e o valor predito pela reta. Em palavras: tudo que o modelo **não** explica.

**Versão sem o erro** (depois de ajustado, para fazer previsão):

```
ŷ = β̂₀ + β̂₁ × x
```

O chapéu (^) indica "estimado". Valores estimados pelo modelo, não os "verdadeiros".

**No nosso projeto.** No notebook 05, ajustamos:

```
nps_predito = 6,63 − 1,02 × delivery_delay_days
```

Onde:
- **β̂₀ = 6,63**: cliente com zero atraso tem NPS predito de 6,63.
- **β̂₁ = −1,02**: cada dia adicional de atraso derruba o NPS predito em 1,02 ponto.

**Snippet.**

```python
# repo/notebooks/05_modelagem_regressao.ipynb
import statsmodels.formula.api as smf
modelo = smf.ols("nps_score ~ delivery_delay_days", data=dados_treino).fit()
print(modelo.summary())
# Coef        Std.Err      t       P>|t|    [0.025   0.975]
# Intercept   6.6280       0.084   78.8    <0.001   6.463    6.793
# delivery    -1.0240      0.039  -26.3    <0.001  -1.101   -0.947
```

**Cuidados.**
- Regressão linear pressupõe **relação linear**. Se a relação real for curva, o modelo vai ser ruim.
- O coeficiente diz a **inclinação média**, não que cada cliente individual obedeça a essa regra.
- Mais sobre premissas na seção 21.

---

## 19. Mínimos quadrados ordinários (OLS)

> **Acrônimo:** **OLS** = *Ordinary Least Squares* (mínimos quadrados ordinários).

**O que é.** Método matemático que **encontra** os melhores valores de β₀ e β₁. O critério é: a reta que **minimiza a soma dos quadrados dos resíduos**.

**Por que "quadrados".**
1. Distâncias positivas (modelo errou para cima) e negativas (errou para baixo) se cancelariam se você só somasse. Elevar ao quadrado torna tudo positivo.
2. O quadrado **penaliza erros grandes** mais do que erros pequenos. Errar 10 conta como 100, errar 1 conta como 1. Isso força a reta a evitar erros grandes.
3. Existe solução fechada (analítica) usando cálculo: derivar e igualar a zero. Isso é raro em ML — quase todo modelo precisa de iteração numérica. OLS tem fórmula direta.

**Matemática.**

Minimizar:
```
Σ (yᵢ − ŷᵢ)² = Σ (yᵢ − (β₀ + β₁xᵢ))²
```

Derivando em relação a β₀ e β₁ e igualando a zero, chega-se a:

```
β̂₁ = Σ((xᵢ − x̄)(yᵢ − ȳ)) / Σ(xᵢ − x̄)²
β̂₀ = ȳ − β̂₁ × x̄
```

Note que o numerador de β̂₁ é a covariância entre x e y; o denominador é a variância de x. Em palavras: **inclinação = covariância / variância**.

**Analogia.** Imagine que você está jogando dardos numa parede e quer marcar o ponto onde "em média" os dardos caíram. Se você só somar as distâncias, dardos à direita anulam dardos à esquerda. Se você somar os **quadrados** das distâncias, todos contam, e o ponto que minimiza essa soma é o "centro de massa" dos dardos.

**Cuidados.**
- OLS é sensível a outliers (porque erros grandes elevados ao quadrado dominam a soma). Robust regression resolve.
- Pressupõe erros independentes e com mesma variância (homocedasticidade — ver seção 25).

---

## 20. R², MAE, MSE, RMSE — métricas de regressão

> **Acrônimos:**
> - **R²** = coeficiente de determinação (não é sigla, mas vou explicar o "ao quadrado")
> - **MAE** = *Mean Absolute Error* (erro médio absoluto)
> - **MSE** = *Mean Squared Error* (erro quadrático médio)
> - **RMSE** = *Root Mean Squared Error* (raiz do erro quadrático médio)

### R²

**O que mede.** Fração da variação da variável Y que o modelo consegue explicar. Vai de 0 (modelo não explica nada, igual chutar a média) a 1 (modelo explica tudo, perfeição irreal).

**Matemática.**

```
R² = 1 − (SS_res / SS_tot)
```

Onde:
- **SS_res** = *Sum of Squares Residual* (soma dos quadrados dos resíduos) = Σ(yᵢ − ŷᵢ)²
- **SS_tot** = *Sum of Squares Total* (soma total dos quadrados) = Σ(yᵢ − ȳ)²
- **ȳ** = média de y

Em palavras: SS_tot mede a variação total; SS_res mede o que sobra depois do modelo. R² é a fração que **deixou** de sobrar.

**O "ao quadrado".** Em regressão linear simples, R² é igual ao quadrado do coeficiente de correlação de Pearson entre x e y. Daí o nome.

**No nosso projeto.** R² no teste = 0,40. Significa que `delivery_delay_days` explica 40% da variação do `nps_score`. Os outros 60% vêm de outras variáveis ou de aleatoriedade.

### MAE

**O que mede.** Erro médio em **unidade da variável** (no nosso caso, pontos de NPS).

**Matemática.**

```
MAE = (1/n) × Σ |yᵢ − ŷᵢ|
```

Onde:
- **|...|** = valor absoluto (positivo, sem sinal)
- **n** = número de observações

**Interpretação.** "Em média, o modelo erra X unidades."

**No nosso projeto.** MAE no teste = 1,58. "Em média, o modelo erra 1,58 ponto de NPS."

### MSE

**O que mede.** Erro médio elevado ao quadrado.

**Matemática.**

```
MSE = (1/n) × Σ (yᵢ − ŷᵢ)²
```

**Interpretação.** Não é diretamente interpretável (unidade ao quadrado), mas é o que OLS minimiza.

### RMSE

**O que mede.** Raiz quadrada do MSE — volta para unidade original.

**Matemática.**

```
RMSE = √MSE
```

**Diferença para MAE.** RMSE penaliza mais erros grandes (porque eleva ao quadrado antes de tirar a raiz). Se você se importa especialmente com não errar feio em casos individuais, use RMSE. Se quer só o erro médio típico, use MAE.

**No nosso projeto.** Reportamos R² e MAE. Não reportamos MSE/RMSE para evitar redundância (RMSE é parecido com MAE e R² já cobre o quadro geral).

**Cuidados.**
- R² infla com mais variáveis. **R² ajustado** corrige isso (penaliza por número de variáveis).
- R² alto não significa modelo bom — pode ser overfitting.
- MAE em unidade da variável é o que executivos entendem.

---

## 21. Resíduos e premissas da regressão linear

**Resíduo.**

```
resíduo = y_real − y_predito = yᵢ − ŷᵢ
```

É a "sobra". O quanto o modelo errou em cada observação.

**Por que olhar resíduos.** Se as premissas da regressão estão sendo respeitadas, os resíduos têm uma assinatura específica: aleatórios, sem padrão, com distribuição aproximadamente normal centrada em zero.

**Premissas (em PT-BR claro).**

1. **Linearidade.** A relação entre x e y é, de fato, linear. Se for curva, o modelo linear vai sistematicamente errar para cima em parte da curva e para baixo na outra parte — e isso aparece no gráfico de resíduos.

2. **Independência dos erros.** Cada resíduo é independente do anterior. Isso falha em séries temporais (resíduo de hoje é parecido com o de ontem). Não temos timestamp no nosso dataset, então OK.

3. **Erros do mesmo tamanho** (homocedasticidade — ver seção 25). A variância dos resíduos é constante ao longo do x.

4. **Normalidade dos erros.** Os resíduos seguem distribuição normal centrada em zero. Necessária para os IC e p-valor dos coeficientes serem válidos. Pelo TCL, com n grande, essa premissa relaxa.

**No nosso projeto.** No notebook 05 (seção 5.6), plotamos:
- **Resíduos vs predito** (painel esquerdo): se vermos um padrão (cone, curva), tem problema. Vimos nuvem aleatória — OK.
- **Q-Q plot dos resíduos** (painel direito, ver seção 22): se os pontos seguem a diagonal, resíduos são normais. Vimos pequeno desvio nas caudas, mas aceitável.

**Snippet.**

```python
# repo/notebooks/05_modelagem_regressao.ipynb
residuos = y_teste - y_pred_teste
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Painel 1: residuos vs predito
axes[0].scatter(y_pred_teste, residuos, alpha=0.3)
axes[0].axhline(0, color="red", linestyle="--")
# Painel 2: Q-Q plot
from scipy import stats
stats.probplot(residuos, dist="norm", plot=axes[1])
```

**Cuidados.**
- Resíduo médio sempre é zero por construção — não precisa testar.
- Se as premissas falham, os coeficientes ainda fazem sentido (são estimativas), mas os IC e p-valor podem estar errados.

---

## 22. Q-Q plot

**Acrônimo:** **Q-Q** = *Quantile-Quantile* (quantil-quantil).

**O que é.** Gráfico que compara os quantis dos seus dados com os quantis de uma distribuição teórica (geralmente normal). Se os dados são normais, os pontos formam uma linha reta na diagonal.

**Como funciona.**
1. Ordene seus dados.
2. Para cada ponto, calcule onde ele "deveria estar" se a distribuição fosse normal (quantil teórico).
3. Plote (quantil teórico, valor real).
4. Se reta = normal. Se curva = não normal.

**Como ler.**
- **Pontos na diagonal:** distribuição é normal. Tudo OK.
- **Caudas para cima** (canto superior direito sobe acima da reta): cauda direita é mais pesada que normal — outliers à direita.
- **Caudas para baixo** (canto inferior esquerdo desce abaixo da reta): cauda esquerda é mais pesada — outliers à esquerda.
- **Curva em S:** distribuição mais "achatada" ou "pontuda" que normal.

**Analogia.** Q-Q plot é tipo um espelho: se sua distribuição é normal, ela "encaixa" perfeitamente no espelho da normal. Onde os pontos saem da diagonal, sua distribuição é diferente.

**No nosso projeto.** Usamos no notebook 05 para conferir normalidade dos resíduos da regressão. Pequeno desvio nas pontas (caudas um pouco mais pesadas) — aceitável dado n grande (TCL ajuda).

**Snippet.**

```python
from scipy import stats
fig, ax = plt.subplots()
stats.probplot(residuos, dist="norm", plot=ax)
ax.set_title("Q-Q plot dos resíduos")
```

---

## 23. Regressão com termo de interação

**O que é.** Quando você suspeita que o efeito de uma variável **depende do valor de outra**, você adiciona um **termo de interação** ao modelo.

**Equação sem interação:**
```
y = β₀ + β₁×x₁ + β₂×x₂ + ε
```

**Equação com interação:**
```
y = β₀ + β₁×x₁ + β₂×x₂ + β₃×(x₁×x₂) + ε
```

Onde:
- **β₃** = coeficiente da interação. Se for diferente de zero (com p-valor pequeno), há efeito moderador.

**Analogia.** Você suspeita que o efeito do exercício na perda de peso depende da idade da pessoa. Sem interação, o modelo diz "exercício reduz X kg, idade reduz Y kg, ponto". Com interação, ele diz "exercício reduz X kg, **mas em pessoas mais velhas reduz só X−algo**".

**No nosso projeto.** H2 testou se a tolerância ao atraso depende do valor do pedido. A hipótese era: clientes com pedidos caros (ticket alto) seriam mais sensíveis a atraso. Em fórmula:

```python
# repo/notebooks/04_eda.ipynb
modelo_com_interacao = smf.ols(
    "nps_score ~ delivery_delay_days * order_value",
    data=dados,
).fit()
# O '*' no statsmodels e' atalho para "x1 + x2 + x1*x2"
```

Resultado: coeficiente da interação com p = 0,58 → não rejeitamos H0 → atraso machuca todos igualmente, independente do ticket → H2 rejeitada.

**Como ler o coeficiente.**

```
nps = β₀ + β₁×atraso + β₂×valor + β₃×(atraso × valor)
    = β₀ + (β₁ + β₃×valor) × atraso + β₂×valor
```

Em palavras: o efeito do atraso sobre o NPS depende do valor — para cada R$1 a mais no pedido, o efeito do atraso muda em β₃.

**Cuidados.**
- Interpretar coeficientes na presença de interação é tricky. Sempre fixe uma das variáveis e veja o efeito da outra.
- Não confunda interação com confundimento (variáveis correlacionadas).

---

## 24. Multicolinearidade

**O que é.** Quando duas (ou mais) variáveis preditoras são fortemente correlacionadas entre si.

**Por que é problema.**
1. Os coeficientes individuais ficam **instáveis** — pequenas mudanças nos dados mudam muito as estimativas.
2. Os p-valores ficam **inflados** — variáveis individualmente "deixam de ser significativas" mesmo sendo no conjunto.
3. **Interpretação fica confusa** — você não sabe se o coeficiente representa o efeito da variável A ou da variável B.

**Analogia.** Você tem dois preditores para previsão do tempo: temperatura em Celsius e temperatura em Fahrenheit. São praticamente a mesma variável transformada. O modelo não consegue distribuir o "crédito" entre as duas — pode dar +1 para Celsius e −0,5 para Fahrenheit, ou vice-versa, ambos "explicando" o mesmo sinal.

**Como detectar.**
- **VIF** (*Variance Inflation Factor*, fator de inflação da variância): se VIF > 10, há multicolinearidade séria.
- **Matriz de correlação entre preditores**: se algum par tem |r| > 0,8, atenção.

**No nosso projeto.** Não enfrentamos diretamente porque escolhemos modelo simples (1 preditor). Mas no modelo agregado do notebook 04 (10+ variáveis) seria importante checar.

**Solução.** Remover uma das variáveis correlacionadas, ou combinar (PCA — *Principal Component Analysis*, análise de componentes principais).

**Cuidados.**
- Multicolinearidade **não afeta** a capacidade preditiva do modelo, só a interpretação dos coeficientes.
- Se você só quer prever (não interpretar), pode até ignorar.

---

## 25. Heterocedasticidade e homocedasticidade

**Acrônimos.** Vêm do grego: *homo* = igual, *hetero* = diferente, *skedastikós* = relativo à dispersão.

- **Homocedasticidade** = variância dos erros constante ao longo do x. Em PT-BR claro: "erros do mesmo tamanho".
- **Heterocedasticidade** = variância dos erros varia com x. Em PT-BR claro: "erros de tamanho diferente".

**Analogia.** Você ajusta uma reta prevendo despesa em função da renda. Nas pessoas de baixa renda, a despesa varia pouco (todo mundo gasta parecido). Nas de alta renda, varia muito (uns gastam pouco, outros muito). O "tamanho" do erro do modelo depende do nível de renda — heterocedasticidade.

**Por que é problema.** Os IC e p-valor dos coeficientes assumem homocedasticidade. Se não for o caso, eles ficam errados (geralmente IC mais estreito do que deveria ser).

**Como detectar.**
- **Visualmente:** gráfico de resíduos vs predito. Se vê um cone (estreito num lado, largo no outro), é hetero.
- **Teste de Breusch-Pagan ou White:** versões formais.

**Solução.**
- **Erros robustos** (HC0, HC1, HC2, HC3 — opções no statsmodels): corrigem os IC sem mudar os coeficientes.
- **Transformar y** (log, raiz): às vezes estabiliza a variância.

**No nosso projeto.** Visualmente, o gráfico de resíduos no notebook 05 não mostrou cone — variância parece OK.

---

## 26. Encoding de variáveis categóricas (one-hot e dummies)

**O que é.** Modelos de regressão só aceitam números. Variável categórica como `customer_region` (Sul, Sudeste, Nordeste) precisa virar números.

### One-hot encoding

Cria uma coluna binária para **cada categoria**. Para 3 regiões, vira 3 colunas:

| customer_region | regiao_Sul | regiao_Sudeste | regiao_Nordeste |
|---|---|---|---|
| Sul | 1 | 0 | 0 |
| Sudeste | 0 | 1 | 0 |
| Nordeste | 0 | 0 | 1 |

### Dummy encoding (regra n−1)

Para regressão, você usa **n−1** colunas, não n. Uma das categorias vira **categoria de referência** (todas as colunas zero):

| customer_region | regiao_Sudeste | regiao_Nordeste |
|---|---|---|
| Sul | 0 | 0 | ← referência (Sul é o "default")
| Sudeste | 1 | 0 |
| Nordeste | 0 | 1 |

**Por que n−1.** Se você usar n colunas, há **multicolinearidade perfeita** (a soma das três colunas é sempre 1, replicando o intercepto). O modelo não consegue resolver. Removendo uma coluna, eliminou o problema.

**Como ler os coeficientes.** Cada coeficiente diz "**quanto a média de y muda quando o cliente é dessa categoria, comparado à categoria de referência**".

**No nosso projeto.** No notebook 04, modelo agregado:

```python
modelo = smf.ols(
    "nps_score ~ delivery_delay_days + C(customer_region)",
    data=dados,
).fit()
# C(...) diz ao statsmodels para tratar como categorica
# Por padrao, ele faz dummy encoding com a primeira categoria como referencia
```

**Cuidados.**
- Para árvores de decisão (Random Forest, XGBoost), você não precisa de one-hot — elas lidam direto com categóricas.
- Se a variável é ordinal (detrator < neutro < promotor), pode usar codificação numérica direta (0, 1, 2).

---

## 27. Matriz de design (design matrix)

**O que é.** A matriz X que statsmodels e sklearn constroem internamente para resolver a regressão.

**Estrutura.** Cada linha é uma observação. Cada coluna é uma variável (incluindo intercepto, dummies, interações):

```
| 1 | atraso | valor | atraso × valor | regiao_Sudeste | regiao_Nordeste |
|---|--------|-------|----------------|----------------|------------------|
| 1 | 0      | 100   | 0              | 1              | 0                |
| 1 | 3      | 250   | 750            | 0              | 1                |
| 1 | 1      | 80    | 80             | 0              | 0                |
```

A primeira coluna de 1's é o intercepto. As outras são as variáveis preditoras (depois de aplicar one-hot, interações, etc.).

**A solução OLS em forma matricial.**

```
β̂ = (XᵀX)⁻¹ Xᵀy
```

Onde:
- **β̂** = vetor de coeficientes estimados
- **X** = matriz de design
- **Xᵀ** = transposta de X
- **(XᵀX)⁻¹** = inversa de XᵀX
- **y** = vetor da variável dependente

Não precisa decorar. O importante: existe **uma fórmula fechada** que resolve OLS em uma operação matricial, daí ele ser tão rápido.

**No nosso projeto.** Quando você escreve `smf.ols("y ~ x1 + x2", data=dados).fit()`, o statsmodels:
1. Constrói a matriz X a partir da fórmula.
2. Resolve a equação acima.
3. Devolve β̂ e os erros padrão.

**Cuidados.**
- Se XᵀX não for inversível (quando há colinearidade perfeita), o método quebra. É por isso que a regra n−1 do dummy encoding existe.

---

---

# Parte V — Classificação

## 28. Classificação binária e threshold

**O que é.** Problema onde a target tem só 2 valores (0/1, sim/não, detrator/não-detrator). O modelo prevê a probabilidade ou um score, e um **threshold** transforma esse número em decisão binária.

**Analogia.** Detector de spam. O modelo dá uma probabilidade de "isto é spam": 0,1 (provavelmente não), 0,9 (provavelmente sim). Você define um corte (threshold = 0,5): acima vira spam, abaixo vira inbox.

**No nosso projeto.** No notebook 06, transformamos a regressão em classificador:

```python
threshold = 6.0
y_pred_det = (nps_predito_teste < threshold).astype(int)
# 1 = predito detrator, 0 = predito nao-detrator
```

**Por que escolher threshold é decisão de produto, não técnica.**

- Threshold baixo → poucos clientes apontados → alta precisão, baixo recall → você só age nos casos óbvios.
- Threshold alto → muitos clientes apontados → baixa precisão, alto recall → você ataca todo mundo, gastando recurso em alarmes falsos.

A escolha depende de **custo dos dois tipos de erro** no negócio.

**Cuidados.**
- Threshold "default" 0,5 (ou no nosso caso, 7) é raramente o melhor. Sempre faça varredura.
- Em datasets desbalanceados (como o nosso, 84% detratores), threshold default pode dar resultado degenerado (ver notebook 06, seção 6.4).

---

## 29. Matriz de confusão

**O que é.** Tabela 2×2 que mostra os 4 possíveis resultados de um classificador binário.

```
                  | Predito: Detrator | Predito: Não-detrator |
Real: Detrator    |        TP         |          FN           |
Real: Não-detrator|        FP         |          TN           |
```

Onde:
- **TP** = *True Positive* (verdadeiro positivo): real detrator, predito detrator. Acerto.
- **TN** = *True Negative* (verdadeiro negativo): real não-detrator, predito não-detrator. Acerto.
- **FP** = *False Positive* (falso positivo): real não-detrator, predito detrator. Alarme falso. Erro Tipo I.
- **FN** = *False Negative* (falso negativo): real detrator, predito não-detrator. Detrator perdido. Erro Tipo II.

**Analogia médica.** Teste de gravidez:
- TP: grávida e teste positivo. ✓
- TN: não grávida e teste negativo. ✓
- FP: não grávida e teste positivo. Susto.
- FN: grávida e teste negativo. Bebê surpresa daqui 9 meses.

**No nosso projeto.** Threshold 6,0 produziu:
- TP = 392 (detratores capturados)
- FP = 46 (alarmes falsos)
- FN = 35 (detratores perdidos)
- TN = 27 (não-detratores corretamente liberados)

**Cuidados.**
- A convenção de qual classe é "positivo" muda entre projetos. No nosso, "positivo" é detrator (a classe que queremos pegar). Em medicina, "positivo" é a doença.
- A matriz de confusão mostra o **balanço** de erros, não só o acerto total.

---

## 30. Acurácia, precisão, recall, F1

Cada métrica responde uma **pergunta de negócio diferente**.

### Acurácia

**Pergunta:** "De todas as previsões, quantas estão certas?"

```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```

**Quando engana.** Em dataset desbalanceado. Se 99% dos clientes são "saudáveis" e 1% têm doença, modelo trivial "saudável para todos" tem 99% de acurácia, mas é inútil.

### Precisão (precision)

**Pergunta:** "Dos que apontei como detrator, quantos eram detrator de verdade?"

```
Precisão = TP / (TP + FP)
```

**Mnemônico.** "Quão preciso é o meu alvo?" Alta precisão = pouco alarme falso.

**Use quando:** alarme falso é caro (acionar SAC custa tempo, dinheiro, irritação).

### Recall (sensibilidade)

**Pergunta:** "Dos detratores reais, quantos eu peguei?"

```
Recall = TP / (TP + FN)
```

**Mnemônico.** "Quanto eu lembro de pegar?" Alto recall = poucos detratores perdidos.

**Use quando:** detrator perdido é caro (cliente que reclama nas redes, churn).

### F1

**Pergunta:** "Tem como combinar precisão e recall num número só?"

```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

**Por que harmônica e não aritmética.** Média harmônica pune valores extremos. Se precisão = 1,0 e recall = 0,1, a aritmética seria 0,55 (parece bom). A harmônica é 0,18 (revela que tem algo errado). F1 só fica alto quando **as duas** são altas.

**No nosso projeto.** Threshold 6,0:
- Acurácia = 83,8% (parece OK, mas baseline trivial é 85,4%, então cuidado)
- Precisão = 89,5% (dos preditos detratores, ~9 em 10 acertou)
- Recall = 91,8% (dos detratores reais, ~9 em 10 capturei)
- F1 = 0,906 (alta combinação)

**Cuidados.**
- F1 dá mesmo peso para precisão e recall. Se você se importa mais com um deles, use **F-beta** (Fβ).
- Em problemas multiclasse, precisão/recall/F1 podem ser calculados por classe ou agregados (macro, micro, weighted).

---

## 31. Curva ROC e AUC (bônus)

> **Acrônimos.**
> - **ROC** = *Receiver Operating Characteristic* (característica de operação do receptor — vem de radar militar de WWII)
> - **AUC** = *Area Under the Curve* (área sob a curva — geralmente sob a ROC)
> - **TPR** = *True Positive Rate* = Recall = TP / (TP + FN)
> - **FPR** = *False Positive Rate* = FP / (FP + TN)

**O que é.** Gráfico que mostra **TPR vs FPR** para todos os thresholds possíveis. Cada ponto é uma escolha de threshold.

- **Diagonal (45°):** classificador aleatório.
- **Canto superior esquerdo:** classificador perfeito (TPR=1, FPR=0).
- **Curva abaulada para o canto:** quanto mais perto do canto, melhor.

**AUC.** Área sob a ROC. Vai de 0,5 (aleatório) a 1,0 (perfeito).
- AUC = 0,5: chute.
- AUC = 0,7: razoável.
- AUC = 0,9+: muito bom.

**Por que NÃO usamos no projeto.** Optamos pela varredura simples de threshold (notebook 06, seção 6.5) por dois motivos:
1. Caía direto no que a aula cobriu (precisão/recall/F1 explícitos).
2. Para o storytelling executivo, "F1 = 0,906" comunica melhor que "AUC = 0,87".

**Quando usar ROC/AUC.**
- Quando você quer comparar dois classificadores de forma agnóstica ao threshold.
- Em datasets relativamente balanceados.
- Em datasets muito desbalanceados, **precision-recall curve** é mais informativa que ROC (vide artigo Saito 2015 nas referências).

---

## 32. Por que não regressão logística

**O que é regressão logística.** Modelo que prevê **probabilidade** de uma classe binária diretamente, usando função sigmoide:

```
P(y = 1 | x) = 1 / (1 + e^(−(β₀ + β₁×x)))
```

A saída fica sempre entre 0 e 1, naturalmente.

**Por que não usamos.**
1. A aula da Fase 1 não cobriu logística. Usá-la sem ter visto seria queimar etapas.
2. Para o problema atual (uma variável dominante, target derivada de escala contínua), a regressão linear simples + threshold dá resultado equivalente a logística com complexidade muito menor.
3. Princípio de **parcimônia**: o modelo mais simples que resolve é o melhor.

**Quando você deveria preferir logística.**
- Quando precisa de probabilidades calibradas (ex.: "este cliente tem 73% de chance de virar detrator").
- Quando há múltiplas variáveis preditoras com efeitos não-lineares mas a target é binária.
- Quando vai integrar com outros modelos que esperam probabilidades.

**Cuidados.**
- Regressão linear pode dar valores fora de [0, 1] se interpretar como probabilidade. No nosso caso, isso não é problema porque interpretamos como NPS predito (0-10), não probabilidade.

---

## 33. Discretização (binning)

**O que é.** Transformar uma variável contínua em categórica, agrupando valores em faixas.

**No nosso projeto.** Usamos em vários lugares:

```python
# Categorizar nps_score em detrator/neutro/promotor
dados["categoria_nps"] = pd.cut(
    dados["nps_score"],
    bins=[-float("inf"), 7, 9, float("inf")],
    labels=["detrator", "neutro", "promotor"],
    right=False,
)

# Faixas de atraso para análise (notebook 05, secao 5.7)
faixa_atraso = pd.cut(
    dados["delivery_delay_days"],
    bins=[-0.5, 0.5, 2.5, 4.5, 100],
    labels=["Sem atraso (0d)", "Atraso leve (1-2d)", "Atraso moderado (3-4d)", "Atraso severo (5+d)"],
)
```

**Quando ajuda.**
- Quando o relacionamento com a target tem ponto de ruptura claro (a NPS canon é um exemplo).
- Quando o público da apresentação entende melhor faixas que números contínuos.
- Quando você quer reduzir efeito de outliers.

**Quando atrapalha.**
- Você **destrói informação** ao binar. Cliente com atraso de 2,1 dias e cliente com atraso de 0,9 dias viram a mesma "faixa leve", mas a diferença é real.
- Reduz capacidade preditiva do modelo.

**Regra prática.**
- Para análise descritiva e apresentação: bina.
- Para modelagem preditiva: mantém contínuo, deixa o modelo descobrir os cortes.

---

---

# Parte VI — Boas práticas de ML

## 34. Train/test split

**O que é.** Separar o dataset em duas partes:
- **Treino:** usado para o modelo "aprender" (ajustar coeficientes).
- **Teste:** usado para avaliar como o modelo se sai em dados que ele **nunca viu**.

**Analogia.** Estudar para a prova. Se você decora todas as questões da lista de exercícios e só faz essas, sabe responder "perfeitamente" ela. Mas na prova de verdade aparecem questões diferentes — e o seu desempenho real só se mede ali.

**Por que separar.** Se você avalia o modelo nos mesmos dados que usou para treinar, está dando uma "avaliação inflada" — o modelo aprendeu **aqueles dados específicos**, incluindo o ruído, e não generaliza.

**Proporção comum.** 80/20 ou 70/30. Usamos 80/20.

**Snippet.**

```python
# repo/notebooks/05_modelagem_regressao.ipynb
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # semente fixa para reprodutibilidade
)
```

**Por que random_state=42.** Convenção da comunidade (referência ao livro "Guia do Mochileiro das Galáxias", onde 42 é "a resposta para a vida, o universo e tudo mais"). Qualquer número fixo funcionaria — o importante é **fixar** para que o split seja sempre o mesmo entre execuções.

**Cuidados.**
- Para classificação com classes desbalanceadas, use `stratify=y` para preservar a proporção em ambos os conjuntos.
- Para séries temporais, **nunca** faça split aleatório — use split temporal (treino antigo, teste recente).
- Veja **data leakage** (seção 39).

---

## 35. Validação cruzada (k-fold)

**O que é.** Generalização do train/test split. Em vez de dividir uma vez (80/20), você divide o dataset em **k pedaços** e faz k rodadas:
- Rodada 1: pedaço 1 é teste, resto é treino. Treina e avalia.
- Rodada 2: pedaço 2 é teste, resto é treino. Treina e avalia.
- ...
- Rodada k: pedaço k é teste, resto é treino. Treina e avalia.

A métrica final é a **média das k avaliações**.

**Analogia.** Em vez de fazer só uma prova, você faz k provas, cada uma cobrindo uma parte diferente da matéria. A média te diz como você sabe a matéria como um todo, e o desvio te diz se você é consistente ou não.

**Por que importa.**
- Dá estimativa mais robusta do desempenho do modelo.
- Reduz dependência do random_state específico do split.

**k mais comum.** k = 5 ou k = 10.

**Por que NÃO usamos no projeto.** Train/test simples já mostrou que o modelo generaliza (R² próximo entre treino e teste). Para um modelo simples e dataset moderado, a sofisticação extra do k-fold não acrescenta muito.

**Snippet (referência).**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    modelo,
    X, y,
    cv=5,             # k = 5 pedaços
    scoring="r2",
)
print(f"R² médio: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Cuidados.**
- k-fold custa k vezes mais computação que train/test simples.
- Para datasets pequenos (n < 100), prefira **leave-one-out** (k = n).

---

## 36. Overfitting e underfitting

### Overfitting

**O que é.** Modelo aprende demais o **ruído específico** do treino, não o padrão geral. Performa muito bem no treino, mal no teste.

**Sintomas.**
- R² alto no treino, R² baixo no teste.
- Modelo muito complexo (muitas variáveis, muitos parâmetros).

**Analogia.** Aluno que decora as respostas da lista mas não entende a matéria. Tira 10 na lista, tira 3 na prova diferente.

### Underfitting

**O que é.** Modelo é simples demais para capturar o padrão. Performa mal em treino e teste.

**Sintomas.**
- R² baixo em ambos.
- Modelo muito simples (poucas variáveis, baixa capacidade).

**Analogia.** Aluno que estudou pouco. Tira 3 em qualquer lugar.

**No nosso projeto.** R² treino ≈ 0,38 e R² teste ≈ 0,40 — praticamente iguais. Não há overfitting (modelo simples demais para overfittar). Pode haver leve underfitting (uma multivariada com mais variáveis chegou a R² = 0,56 no notebook 04). Mas o trade-off de complexidade não vale a pena para o objetivo do projeto.

**Soluções para overfitting.**
- Reduzir complexidade (menos variáveis, modelo mais simples).
- Regularização (L1, L2 — penalizam coeficientes grandes).
- Mais dados de treino.
- Validação cruzada para detectar.

**Soluções para underfitting.**
- Aumentar complexidade (mais variáveis, modelo mais sofisticado).
- Adicionar interações, transformações não-lineares.

---

## 37. Feature engineering vs feature selection

### Feature engineering

**O que é.** Criar novas variáveis a partir das existentes, baseado em conhecimento do domínio.

**Exemplos.**
- Combinar duas colunas: `valor_por_item = order_value / items_quantity`.
- Aplicar transformação: `log_order_value = log(1 + order_value)`.
- Extrair de data: `mes_compra` a partir de `data_pedido`.
- Categorizar: `faixa_atraso` a partir de `delivery_delay_days`.

**No nosso projeto.** No notebook 03 (`features.py`):

```python
# Categorizacao da target
dados["categoria_nps"] = pd.cut(...)

# Transformacao log para reduzir assimetria
dados["log_order_value"] = np.log1p(dados["order_value"])
```

### Feature selection

**O que é.** Escolher **quais** das variáveis disponíveis usar no modelo.

**Métodos.**
- **Filtros:** correlação com target, variância, qui-quadrado. Rápido, ignora interações.
- **Wrappers:** treina modelo com subconjuntos diferentes. Caro, mas considera interações.
- **Embedded:** o modelo escolhe sozinho durante o treinamento (LASSO, árvores).

**No nosso projeto.** Usamos **seleção por hipótese**: cada variável só entra no modelo se passar em um teste estatístico no notebook 04. Resultado: usamos só `delivery_delay_days`.

**Diferença.** Engineering **cria** features. Selection **escolhe** features. Engineering vem antes; selection depois.

---

## 38. Pipelines do sklearn

**O que é.** Estrutura do sklearn que encadeia múltiplos passos de transformação + modelagem em um único objeto.

**Por que importa.**
- Garante que **mesmas transformações** sejam aplicadas em treino e teste.
- Evita data leakage (seção 39) por aplicar transformações **dentro** do split de validação.
- Facilita serialização e deploy.

**Exemplo (referência, não usamos no projeto).**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),       # passo 1: padroniza
    ("modelo", LinearRegression()),     # passo 2: modela
])

pipeline.fit(X_treino, y_treino)        # treina os dois passos juntos
pipeline.predict(X_teste)               # aplica os dois passos no teste
```

**Por que NÃO usamos.** Modelo simples (1 variável, sem pré-processamento). Pipeline seria overkill. Em projetos com muitas transformações, pipeline é essencial.

---

## 39. Data leakage

**O que é.** Quando informação do **teste** vaza para o **treino** durante o pipeline, inflando artificialmente o desempenho.

**Exemplos clássicos.**
- Calcular média/desvio para padronização **antes** do split. A média do dataset inteiro inclui o teste, então o treino "viu" o teste.
- Usar variáveis que só existem **depois** do evento que você quer prever. Ex.: prever churn usando "data de cancelamento".
- Duplicatas entre treino e teste.

**Como evitar.**
- Sempre faça split **primeiro**, depois pré-processe.
- Use Pipeline do sklearn (seção 38).
- Reflita sobre o **timing** das variáveis: estavam disponíveis no momento da decisão?

**No nosso projeto.** A target `categoria_nps` é derivada de `nps_score`. Se você incluísse `nps_score` como preditor, seria leakage perfeito (modelo "preveria" a categoria com 100% de acurácia, porque está prevendo algo derivado do próprio preditor). Por isso, no notebook 06, X só inclui `delivery_delay_days`.

**Cuidados.**
- Leakage é silencioso. Resultado parece bom, mas o modelo falha em produção.
- Em séries temporais, leakage é especialmente comum — sempre confira se preditor está disponível **antes** do evento.

---

---

# Parte VII — Ecossistema Python e reprodutibilidade



---

*Caderno em construção. Demais seções do índice serão preenchidas conforme o estudo avança.*
