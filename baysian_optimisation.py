import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, make_scorer
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import time

# Définir votre fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    average_recall = (recall_class_1 + recall_class_0) / 2
    return average_recall

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

# Définir l'espace des hyperparamètres pour l'optimisation bayésienne
param_dist = {
    'n_estimators': (10, 500),
    'max_depth': (3, 50),
    'min_samples_split': (2, 50),
    'min_samples_leaf': (1, 50),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
}

# Configuration de la recherche bayésienne avec custom_precision et accuracy
custom_scorer = make_scorer(custom_precision)
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_dist,
    n_iter=150,
    cv=5,  # Validation croisée à 5 plis
    verbose=3,
    n_jobs=-1,  # Utiliser tous les cœurs disponibles
    random_state=42,
    scoring={'custom_precision': custom_scorer, 'accuracy': 'accuracy'},  # Calculer les deux métriques
    refit='custom_precision',  # Optimiser sur custom_precision
    return_train_score=True  # Pour récupérer les scores d'entraînement
)

# Démarrer le chronomètre pour BayesSearchCV
start_time = time.time()

# Exécuter BayesSearchCV pour l'optimisation
bayes_search.fit(X_train, y_train)

# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps d'exécution de BayesSearchCV : {elapsed_time / 60:.2f} minutes")

# Meilleurs paramètres trouvés
best_params = bayes_search.best_params_
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

# Extraire les résultats du BayesSearchCV
results = bayes_search.cv_results_

# Extraire les scores moyens d'accuracy pour chaque combinaison d'hyperparamètres
mean_test_accuracy = results['mean_test_accuracy']

# Afficher l'évolution des scores d'accuracy (en fonction des itérations)
plt.figure(figsize=(10, 6))
plt.plot(mean_test_accuracy, marker='o', color='blue', linestyle='-', label='Accuracy moyenne de validation')
plt.xlabel('Itérations (combinaisons d\'hyperparamètres)')
plt.ylabel('Accuracy Moyenne de Validation')
plt.title('Évolution de l\'Accuracy lors de l\'Optimisation des Hyperparamètres (BayesSearchCV)')
plt.grid(True)
plt.legend()
plt.show()