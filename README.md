# Predição de Doença Renal Crônica com Aprendizado de Máquina

Este repositório contém o código-fonte e materiais relacionados ao estudo **"Aplicação de Aprendizado de Máquina no Auxílio ao Diagnóstico de Doenças Renais Crônicas"**, cujo objetivo é comparar diferentes algoritmos de aprendizado supervisionado para identificar casos de Doença Renal Crônica (CKD) a partir de dados clínicos.

## 📌 Objetivo
O projeto tem como meta analisar e comparar o desempenho de quatro algoritmos de classificação para prever a presença de CKD, contribuindo para a detecção precoce e suporte à decisão clínica.

## 🧠 Algoritmos Utilizados
- **K-Nearest Neighbors (KNN)**
- **Decision Tree (DT)**
- **Random Forest (RF)**
- **Fuzzy Unordered Rule Induction Algorithm (FURIA)**
    > ℹ️ *Os dados de desempenho e as regras fuzzy do algoritmo FURIA foram extraídos com suporte da ferramenta [WEKA](https://www.cs.waikato.ac.nz/ml/weka/), utilizada para execução e análise do modelo.*

## 📊 Dataset
O conjunto de dados utilizado foi o **Chronic Kidney Disease Dataset** disponível no [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease).

- **Instâncias:** 400
- **Atributos:** 24 (clínicos e laboratoriais)
- **Classes:** CKD e NotCKD
- **Desbalanceamento:** ~62,5% CKD e 37,5% NotCKD

## 🔄 Pré-processamento
O pré-processamento incluiu:
1. **Tratamento de valores ausentes**: imputação por média (atributos numéricos) e moda (atributos categóricos).
2. **Codificação de variáveis categóricas**: `Label Encoding`.
3. **Normalização**: `Min-Max Scaling`.
4. **Divisão treino/teste**: 80/20, estratificada.

## 📈 Métricas de Avaliação
Para cada algoritmo, foram calculadas:
- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **Matriz de Confusão**

