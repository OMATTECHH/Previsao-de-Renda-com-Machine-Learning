import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar os dados
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status',
           'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
           'Hours per week', 'Native Country', 'Income']

df = pd.read_csv(data_url, names=columns, na_values=' ?', skipinitialspace=True)

# Remover valores nulos
df.dropna(inplace=True)

# Converter variáveis categóricas para numéricas
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Selecionar features e target
X = df.drop('Income', axis=1)
y = df['Income']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo KNN: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
