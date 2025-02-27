import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, make_scorer
from scipy.stats import randint, uniform
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

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir l'espace de recherche des hyperparamètres avec des intervalles
param_dist = {
    'n_estimators': randint(10, 201),  # Intervalles pour n_estimators entre 10 et 200
    'max_depth': randint(3, 26),       # Intervalles pour max_depth entre 3 et 25
    'min_samples_split': randint(2, 16),  # Intervalles pour min_samples_split entre 2 et 15
    'min_samples_leaf': randint(1, 7),   # Intervalles pour min_samples_leaf entre 1 et 6
    'max_features': ["sqrt", "log2"],   # Liste explicite pour max_features
    'criterion': ["gini", "entropy"]     # Liste explicite pour criterion
}

# Configuration de la recherche aléatoire
custom_scorer = make_scorer(custom_precision)
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=500,  # Nombre d'itérations pour RandomizedSearchCV
    scoring=custom_scorer,
    cv=5,  # Validation croisée à 5 plis
    n_jobs=-1,  # Utiliser tous les cœurs disponibles
    verbose=2,  # Afficher les détails de l'exécution
    random_state=42,
    return_train_score=True
)

# Démarrer le chronomètre
start_time = time.time()

# Exécuter la recherche aléatoire
random_search.fit(X_train, y_train)

# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps d'exécution total : {elapsed_time / 60:.2f} minutes")

# Meilleurs paramètres trouvés
best_params = random_search.best_params_
print("Meilleurs paramètres après optimisation :", best_params)

# Entraîner le modèle optimisé avec les meilleurs paramètres
best_model = random_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

# Calculer la précision personnalisée et l'accuracy sur l'ensemble de test
precision_optimized = custom_precision(y_test, y_pred_optimized)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

print(f"Précision personnalisée après optimisation : {precision_optimized:.4f}")

# Graphique de l'évolution du score moyen basé sur custom_precision
results_df = pd.DataFrame(random_search.cv_results_)
mean_test_scores = results_df['mean_test_score']  # Scores moyens (basés sur custom_precision)
param_combinations = range(1, len(mean_test_scores) + 1)  # Numéro de combinaison

plt.figure(figsize=(8, 6))
plt.plot(param_combinations, mean_test_scores, marker='o', linestyle='-', color='blue')
plt.xlabel('Combinaison de paramètres')
plt.ylabel('Score moyen (Précision Personnalisée)')
plt.title('Évolution de la précision personnalisée pour chaque combinaison de paramètres')
plt.grid(True)
plt.show()