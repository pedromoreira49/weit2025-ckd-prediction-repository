# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn import tree

# Carregamento do dataset
df = pd.read_csv('kidney_disease.csv')

# Removendo a coluna 'id', que não contribui para o modelo
df.drop('id', axis=1, inplace=True)

# Renomeando as colunas para nomes mais legíveis e consistentes
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

# Removendo algumas colunas não utilizadas no estudo
df.drop('age', axis=1, inplace=True)
df.drop('red_blood_cells', axis=1, inplace=True)
df.drop('pus_cell', axis=1, inplace=True)

# Convertendo colunas numéricas que foram lidas como string
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

# Separando colunas categóricas e numéricas
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# Inspecionando valores únicos nas variáveis categóricas
for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")

# Tratando valores inconsistentes nas variáveis categóricas
df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')
df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})

# Convertendo a variável alvo para binário (0 = CKD, 1 = Not CKD)
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors='coerce')

# Conferindo valores únicos nas colunas-alvo de correção
cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']
for col in cols:
    print(f"{col} has {df[col].unique()} values\n")

for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")

# Checando valores ausentes
df.isna().sum().sort_values(ascending = False)
df[num_cols].isnull().sum()
df[cat_cols].isnull().sum()

# Função para imputação de valores numéricos ausentes usando amostras aleatórias
def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

# Função para imputação de valores categóricos usando a moda
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

# Aplicando imputação nos atributos numéricos
for col in num_cols:
    random_value_imputation(col)

df[num_cols].isnull().sum()

# Aplicando imputação nos atributos categóricos
for col in cat_cols:
    impute_mode(col)

# Removendo registros com valores nulos restantes nas variáveis categóricas
df.dropna(subset=cat_cols, inplace=True)

# Preenchendo valores categóricos ausentes com 'Unknown' (se houver)
for col in cat_cols:
    df[col].fillna('Unknown', inplace=True)

df[cat_cols].isnull().sum()

for col in cat_cols:
    print(f"{col} has {df[col].nunique()} categories\n")

# Convertendo variáveis categóricas para numéricas com Label Encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Definindo listas de variáveis numéricas e binárias
numeric_cols = [
    'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine',
    'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'red_blood_cell_count'
]
binary_cols = [
    'pus_cell_clumps', 'bacteria', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'
]

# Criando cópia do DataFrame para normalização (usado no FURIA)
df_fuzzy = df.copy()

# Aplicar normalização Min-Max nas colunas numéricas
scaler = MinMaxScaler()
df_fuzzy[numeric_cols] = scaler.fit_transform(df_fuzzy[numeric_cols])

# Converter colunas binárias para float
df_fuzzy[binary_cols] = df_fuzzy[binary_cols].astype(float)

# Separando variáveis independentes (X) e alvo (y)
X = df_fuzzy.drop('class', axis = 1)
y = df_fuzzy['class']

# Dividindo em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ====== MODELOS ======

# Modelo 1: Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of decision tree

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(X_test))}")

# Modelo 2: Random Forest
rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rd_clf.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(X_test))}")

# Modelo 3: KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of knn

knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, knn.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, knn.predict(X_test))}")

