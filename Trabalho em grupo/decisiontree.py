# Importação das bibliotecas necessárias
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# Carregar os dados
data = pd.read_csv('C:/Users/ferre/Documents/GitHub/Inteligência Artificial/Trabalho em grupo/heart-disease.csv')

# Separar as features (X) e o alvo (y)
X = data.drop('target', axis=1)  # 'target' representa a coluna de destino
y = data['target']

# Pré-processamento: Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializando o modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42)

# Validação Cruzada com 5 folds
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
print(f"Acurácia média (cross-validation): {cv_scores.mean()}")

# GridSearchCV para encontrar os melhores hiperparâmetros
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")

# Usando o melhor modelo encontrado pelo GridSearchCV
best_clf = grid_search.best_estimator_

# Treinando o modelo com os dados de treino
best_clf.fit(X_train, y_train)

# Realizando previsões com os dados de teste
y_pred = best_clf.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Acurácia (melhor modelo): {accuracy}")
print("Matriz de Confusão (melhor modelo):")
print(conf_matrix)
print("\nRelatório de Classificação:")
print(classification_rep)
print(f"AUC-ROC: {auc_roc}")

# Visualização da árvore de decisão
plt.figure(figsize=(20, 10))
tree.plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.show()
