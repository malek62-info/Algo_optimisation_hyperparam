import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, make_scorer
import matplotlib.pyplot as plt
import time

# Fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Charger les données
file_path = 'C:/tuning/data_cleaned.xlsx'
df = pd.read_excel(file_path)

# Sélection des variables et de la cible
X = df.drop(columns=['Survived'])
y = df['Survived']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ["sqrt", "log2"],
    'criterion': ["gini", "entropy"]
}

# Configuration de la recherche par grille
custom_scorer = make_scorer(custom_precision)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # Validation croisée à 5 plis
    n_jobs=-1,  # Utiliser tous les cœurs disponibles
    verbose=2  # Afficher les détails de l'exécution
)

# Démarrer le chronomètre
start_time = time.time()

# Exécuter GridSearchCV sans limite de temps
grid_search.fit(X_train, y_train)

# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps d'exécution total : {elapsed_time / 60:.2f} minutes")

# Meilleurs paramètres trouvés
best_params = grid_search.best_params_
print("Meilleurs paramètres après optimisation :", best_params)

# Graphique de l'évolution du score moyen
results_df = pd.DataFrame(grid_search.cv_results_)
mean_test_scores = results_df['mean_test_score']  # Scores moyens pour chaque combinaison
param_combinations = range(1, len(mean_test_scores) + 1)  # Numéro de combinaison

plt.figure(figsize=(8, 6))
plt.plot(param_combinations, mean_test_scores, marker='o', linestyle='-')
plt.xlabel('Combinaison de paramètres')
plt.ylabel('Score moyen (Précision Personnalisée)')
plt.title('Évolution de la précision personnalisée pour chaque combinaison de paramètres')
plt.grid(True)
plt.show()