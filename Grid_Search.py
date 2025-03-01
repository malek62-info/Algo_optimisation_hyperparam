import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, make_scorer
import time
import matplotlib.pyplot as plt

# Définition de la métrique personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Chemins des fichiers
data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_train_t3.xlsx"
data_test_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_test_t3.xlsx"

# Charger les données d'entraînement et de test
df_train = pd.read_excel(data_train_path)
df_test = pd.read_excel(data_test_path)

# Sélection des variables et de la cible
X_train = df_train.drop(columns=['Survived'])
y_train = df_train['Survived']
X_test = df_test.drop(columns=['Survived'])
y_test = df_test['Survived']

# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [10, 30, 40, 51],
    'max_depth': [3, 5, 6, 7],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1],
    'max_features': ["sqrt", "log2"],
    'criterion': ["gini", "entropy"]
}

# Configuration de la recherche par grille avec custom_precision
custom_scorer = make_scorer(custom_precision)  # Utilisation de la métrique personnalisée
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # Validation croisée à 5 plis
    n_jobs=-1,  # Utiliser tous les cœurs disponibles
    verbose=2,  # Afficher les détails de l'exécution
    scoring=custom_scorer  # Optimiser sur custom_precision
)

# Démarrer le chronomètre pour GridSearchCV
start_time = time.time()

# Exécuter GridSearchCV pour l'optimisation
grid_search.fit(X_train, y_train)

# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps d'exécution de GridSearchCV : {elapsed_time / 60:.2f} minutes")

# Meilleurs paramètres trouvés
best_params = grid_search.best_params_
print("Meilleurs paramètres après optimisation :", best_params)

# Créer et entraîner le modèle avec les meilleurs paramètres
optimized_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    criterion=best_params['criterion'],
    random_state=42
)

# Entraîner le modèle
optimized_model.fit(X_train, y_train)

# Prédictions sur l'ensemble d'entraînement et de test
y_pred_train = optimized_model.predict(X_train)
y_pred_test = optimized_model.predict(X_test)

# Calcul des scores avec la métrique personnalisée et l'accuracy
custom_score_train = custom_precision(y_train, y_pred_train)
custom_score_test = custom_precision(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Affichage des résultats avec custom_precision et accuracy
print("\nRésultats :")
print(f"Score personnalisé sur l'ensemble d'entraînement : {custom_score_train:.4f}")
print(f"Accuracy sur l'ensemble d'entraînement : {accuracy_train:.4f}")
print(f"Score personnalisé sur l'ensemble de test : {custom_score_test:.4f}")
print(f"Accuracy sur l'ensemble de test : {accuracy_test:.4f}")

# Extraire les résultats du GridSearchCV
results = grid_search.cv_results_

# Extraire les scores moyens pour chaque combinaison d'hyperparamètres
mean_test_scores = results['mean_test_score']

# Afficher l'évolution des scores (en fonction des itérations)
plt.figure(figsize=(10, 6))
plt.plot(mean_test_scores, marker='o', color='blue', linestyle='-', label='Score moyen de validation')

plt.xlabel('Itérations (combinations d\'hyperparamètres)')
plt.ylabel('Score Moyen de Validation')
plt.title('Évolution de la Performance lors de l\'Optimisation des Hyperparamètres')
plt.grid(True)
plt.legend()
plt.show()
