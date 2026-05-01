# Referências de estudo — Tech Challenge Fase 1 (NPS Preditivo)

Lista enxuta de materiais para você aprofundar todo o conteúdo abordado no projeto. Critério de seleção: livros baratos disponíveis na Amazon Brasil, artigos canônicos gratuitos, vídeos curtos com bom didatismo (não cursos de 40h) e tutoriais oficiais bem mantidos.

> **Como usar este arquivo.** A ordem das seções segue a curva natural de aprendizado: comece pela base estatística, depois EDA, depois modelagem. Se você já tem familiaridade com algum bloco (ex.: Python para DS, por já programar), pule e use só como consulta.

---

## 1. Estatística e probabilidade (a base de tudo)

### Livros

- **"Naked Statistics: Stripping the Dread from the Data"** — Charles Wheelan
  - Custo: ~R$ 60-80 ([buscar na Amazon BR](https://www.amazon.com.br/s?k=naked+statistics+charles+wheelan))
  - Por que: o livro mais acessível de estatística que existe, escrito como crônica jornalística. Cobre média, mediana, distribuição, correlação, inferência, teste de hipótese, regressão. Sem fórmulas pesadas, mas com intuição cirúrgica.
  - O que esperar: lê em 10-15 dias em ritmo confortável. É o "primeiro livro" para quem não vem de exatas.

- **"Estatística Básica"** — Wilton Bussab e Pedro Morettin
  - Custo: ~R$ 130-150 ([buscar na Amazon BR](https://www.amazon.com.br/s?k=estatistica+basica+bussab+morettin))
  - Por que: o clássico brasileiro usado em todas as graduações de exatas. Mais formal que o Wheelan, com exercícios. Cobre tudo do nosso projeto com rigor matemático.
  - O que esperar: livro de consulta para o resto da vida. Não precisa ler do início ao fim — vai capítulo a capítulo conforme o tema apareça.

### Vídeos

- **StatQuest com Josh Starmer** — [youtube.com/@statquest](https://www.youtube.com/@statquest) (gratuito)
  - Por que: o melhor canal de estatística e ML do YouTube. Cada conceito tem um vídeo de 5-15 minutos, com analogias visuais e a frase de abertura "BAM!" entre as ideias.
  - Playlists chave para o nosso projeto:
    - "Statistics Fundamentals" — média, variância, distribuição, p-valor
    - "Linear Regression and Linear Models" — tudo do notebook 05
    - "Machine Learning Fundamentals" — métricas, F1, confusion matrix

- **3Blue1Brown** — [youtube.com/@3blue1brown](https://www.youtube.com/@3blue1brown) (gratuito)
  - Por que: animações matemáticas excelentes. A série "Essence of Calculus" e "Essence of Linear Algebra" dá a intuição visual que fórmulas não dão.
  - O que ver: a aula sobre derivadas (necessária para entender mínimos quadrados) e a aula sobre Teorema de Bayes.

### Curso interativo

- **Khan Academy — Statistics and Probability** — [khanacademy.org/math/statistics-probability](https://www.khanacademy.org/math/statistics-probability) (gratuito, com legendas em PT-BR)
  - Por que: aulas curtas e exercícios automáticos. Cobre desde média/mediana até regressão e teste de hipótese.
  - O que esperar: 30-50 horas para fazer o curso completo. Ideal para quem aprende fazendo.

---

## 2. NPS e Customer Experience

- **"The One Number You Need to Grow"** — Frederick Reichheld (Harvard Business Review, 2003)
  - [hbr.org/2003/12/the-one-number-you-need-to-grow](https://hbr.org/2003/12/the-one-number-you-need-to-grow) (~US$ 9 ou gratuito com cadastro de testes)
  - Por que: o paper original que criou o NPS. Lê em 30 minutos. Entende a lógica por trás do indicador, por que neutros não contam, e como o NPS foi vendido para o mercado.

- **Net Promoter System (Bain)** — [netpromotersystem.com](https://www.netpromotersystem.com/)
  - Por que: site oficial mantido pela Bain, criadora do NPS. Tem case studies, FAQ e a evolução do NPS para "NPS 3.0" (Earned Growth Rate).
  - O que ver: as seções "What is NPS?" e "Why NPS Matters" para complementar o paper original.

- **Crítica acadêmica do NPS** — Keiningham, Cooil, Andreassen e Aksoy (2007)
  - Buscar no Google Scholar: ["A Longitudinal Examination of Net Promoter and Firm Revenue Growth"](https://scholar.google.com/scholar?q=A+Longitudinal+Examination+of+Net+Promoter+and+Firm+Revenue+Growth)
  - Por que: contraponto científico ao paper de Reichheld. Mostra que NPS não é tão preditivo de crescimento quanto Reichheld afirmou. Importante para ter visão balanceada.

---

## 3. EDA e visualização de dados

### Livros

- **"Storytelling com Dados: Um Guia Sobre Visualização de Dados para Profissionais de Negócios"** — Cole Nussbaumer Knaflic
  - Custo: ~R$ 70-90 ([buscar na Amazon BR](https://www.amazon.com.br/s?k=storytelling+com+dados+cole+knaflic))
  - Por que: o livro de visualização para apresentações executivas. Os 6 princípios do Knaflic (entender contexto, escolher gráfico certo, eliminar ruído, focar atenção, pensar como designer, contar história) estão em todos os nossos slides.
  - O que esperar: leitura curta (1 fim de semana). Você vai começar a "doer" ao ver gráficos ruins na empresa depois.

### Vídeos e tutoriais

- **Tutorial oficial do seaborn** — [seaborn.pydata.org](https://seaborn.pydata.org/) (gratuito)
  - Por que: a galeria oficial mostra cada tipo de gráfico estatístico com código mínimo. Use como referência rápida.

- **Documentação do matplotlib — Quick Start** — [matplotlib.org/stable/users/explain/quick_start.html](https://matplotlib.org/stable/users/explain/quick_start.html) (gratuito)
  - Por que: explica a anatomia de uma figura matplotlib (Figure, Axes, Artist) que confunde todo iniciante. Lê em 20 minutos.

---

## 4. Regressão linear (matemática + intuição)

- **"An Introduction to Statistical Learning" (ISLR)** — James, Witten, Hastie, Tibshirani
  - [statlearning.com](https://www.statlearning.com/) (gratuito, PDF oficial)
  - Por que: o livro padrão de ML estatístico no mundo acadêmico. Capítulo 3 (Linear Regression) é a referência canônica para tudo que fizemos no notebook 05. Capítulo 4 (Classification) cobre o que evoluiremos no notebook 06 quando trocarmos threshold por regressão logística.
  - O que esperar: matemática presente, mas sempre acompanhada de explicação. Tem exercícios em R e Python.

- **StatQuest — playlist Linear Regression** — buscar "StatQuest Linear Regression" no YouTube
  - Por que: 4-5 vídeos curtos cobrindo OLS, R², resíduos. Complemento visual ao capítulo 3 do ISLR.

- **"Naked Statistics", capítulos 11 e 12** (regressão e regressão múltipla)
  - Mesmo livro da seção 1. Os capítulos sobre regressão são particularmente bons.

---

## 5. Classificação e métricas (precisão, recall, F1)

- **"The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** — Saito e Rehmsmeier (PLOS ONE, 2015)
  - [journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) (gratuito)
  - Por que: explica por que F1 e precision/recall são mais informativos que acurácia em datasets desbalanceados — exatamente o caso da nossa base com 84% detratores.

- **StatQuest — playlist Confusion Matrix and ROC** — buscar no YouTube
  - Por que: vídeos de 8-12 minutos sobre matriz de confusão, sensibilidade/especificidade, ROC, AUC. Visualizações claras de cada métrica.

---

## 6. CRISP-DM e metodologia de projeto

- **CRISP-DM 1.0 — Step-by-Step Data Mining Guide** — IBM/SPSS (1999, ainda atual)
  - Buscar "CRISP-DM 1.0 PDF" no Google — vários mirrors gratuitos
  - Por que: documento original de 76 páginas que define as 6 fases do CRISP-DM em detalhe. É a fonte que você cita em entrevistas/provas.

- **"What is CRISP-DM?"** — Data Science PM
  - [datascience-pm.com/crisp-dm-2/](https://www.datascience-pm.com/crisp-dm-2/)
  - Por que: artigo curto que resume CRISP-DM, suas críticas e alternativas modernas (ML Lifecycle, TDSP, MLOps). Bom para contextualizar.

- **Cookiecutter Data Science** — [cookiecutter-data-science.drivendata.org](https://cookiecutter-data-science.drivendata.org/)
  - Por que: o template de pasta que usamos no `repo/`. A documentação explica o "porquê" de cada pasta e convenção.

---

## 7. Python para Data Science

### Livros

- **"Python para Análise de Dados"** — Wes McKinney (3ª edição, 2022)
  - Custo: ~R$ 120-150 ([buscar na Amazon BR](https://www.amazon.com.br/s?k=python+para+analise+de+dados+mckinney))
  - Versão online gratuita: [wesmckinney.com/book/](https://wesmckinney.com/book/)
  - Por que: McKinney é o criador do pandas. O livro é a referência canônica para pandas, NumPy e análise de dados em Python. Cobre 90% do que usamos no nosso projeto.
  - O que esperar: 500 páginas, mas é livro de consulta. Você não lê linear — vai ao capítulo do tópico que precisa.

### Tutoriais oficiais

- **Pandas — User Guide** — [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) (gratuito)
  - Especialmente: "10 minutes to pandas", "Group By: split-apply-combine", "Categorical data".

- **NumPy — Documentation** — [numpy.org/doc](https://numpy.org/doc/) (gratuito)
  - Especialmente: "NumPy: the absolute basics for beginners".

### Curso interativo

- **Kaggle Learn** — [kaggle.com/learn](https://www.kaggle.com/learn) (gratuito)
  - Por que: micro-cursos de 3-5 horas cada. Os mais relevantes: "Pandas", "Data Visualization", "Intro to Machine Learning". Você faz o exercício no próprio Kaggle (rode notebook gratuito na nuvem) e ganha certificado.

---

## 8. Engenharia de software para Data Science

- **"Hypermodern Python"** — Claudio Jolowicz (série de posts)
  - [cjolowicz.github.io/posts/hypermodern-python-01-setup](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) (gratuito)
  - Por que: série de 6 posts explicando como montar um projeto Python moderno, com gestão de dependências, testes, lint, type checking. Você vai reconhecer várias decisões do nosso projeto (`uv`, `ruff`, `pytest`).

- **Documentação do uv** — [docs.astral.sh/uv](https://docs.astral.sh/uv/)
  - Por que: o gerenciador de pacotes que usamos. Substitui pip + virtualenv + pip-tools com um único binário escrito em Rust. Vale ler a página "Why uv?" para entender o problema que ele resolve.

---

## 9. Bibliotecas de modelagem

- **scikit-learn — User Guide** — [scikit-learn.org/stable](https://scikit-learn.org/stable/)
  - Especialmente: "Supervised learning" → "Linear Models", e "Model selection and evaluation" → "Metrics".
  - Por que: a biblioteca padrão de ML em Python. Mesmo usando statsmodels para inferência, sklearn é o que vai aparecer em qualquer vaga.

- **statsmodels — Documentation** — [statsmodels.org](https://www.statsmodels.org/)
  - Especialmente: "Linear Regression" e "Statistics" (testes de hipótese).
  - Por que: a biblioteca para inferência estatística clássica em Python. É o que dá tabelas com p-valor e IC, igual a R.

---

## 10. Caminho avançado (depois deste projeto)

Quando você sentir que dominou o projeto atual, estes são os próximos passos naturais:

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** — Aurélien Géron (3ª edição, O'Reilly)
  - Repositório oficial com notebooks: [github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)
  - Por que: o caminho natural depois de regressão linear → árvores → ensembles → redes neurais. Custo: ~R$ 200-250 (vale cada centavo).

- **"R for Data Science"** — Hadley Wickham e Garrett Grolemund
  - [r4ds.had.co.nz](https://r4ds.had.co.nz/) (gratuito, em inglês)
  - Por que: mesmo trabalhando em Python, ler este livro melhora seu raciocínio analítico. Wickham é o criador do tidyverse e a forma como ele pensa sobre "tidy data" virou padrão na comunidade.

---

## Ordem sugerida para o seu caso

Você é engenheiro de plataforma sênior (não vem de exatas formal). Sugestão de ordem prática:

1. **Naked Statistics** (Wheelan) — 2 semanas, ritmo confortável
2. **StatQuest** (vídeos avulsos) — em paralelo, conforme um conceito específico aparecer
3. **ISLR cap. 3** (regressão linear) — 1 semana com lápis e papel
4. **Storytelling com Dados** (Knaflic) — 1 fim de semana
5. **Wes McKinney** (livro pandas) — referência permanente, vai capítulo a capítulo conforme você usa pandas no dia a dia
6. **Hypermodern Python** (Jolowicz) — 1 tarde de fim de semana
7. **Hands-On ML** (Géron) — 2-3 meses, é o livro definitivo para a Fase 2 do MBA
