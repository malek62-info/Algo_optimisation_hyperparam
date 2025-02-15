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
file_path = 'C:/irace_random_forest/data_cleaned.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Fichier {file_path} non trouvé. Assurez-vous qu'il est présent dans le répertoire actuel.")
    exit()

# Sélection des variables et de la cible
X = df.drop(columns=['Survived'])
y = df['Survived']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle avec des hyperparamètres par défaut (précision initiale)
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
precision_initial = custom_precision(y_test, y_pred_default)
print(f"Précision avec hyperparamètres par défaut : {precision_initial:.4f}")

# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'max_depth': [3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50],
    'min_samples_leaf': [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50],
    'max_features': ["sqrt", "log2", None],
    'criterion': ["gini", "entropy"]
}

# Nombre total de combinaisons
total_combinations = np.prod([len(values) for values in param_grid.values()])
print(f"Nombre total de combinaisons possibles : {total_combinations}")

# Configuration de la recherche par grille
custom_scorer = make_scorer(custom_precision)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # Réduit pour accélérer
    n_jobs=-1,
    verbose=2,
    scoring=custom_scorer
)

# Limite de temps d'exécution (2 heures)
total_time = 2 * 3600  # 2 heures en secondes
start_time = time.time()

# Exécuter GridSearchCV avec une limite de temps
try:
    grid_search.fit(X_train, y_train)
except KeyboardInterrupt:
    print("Temps écoulé : 2 heures, arrêt de la recherche.")

# Temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps d'exécution : {elapsed_time / 60:.2f} minutes")

# Meilleurs paramètres trouvés
best_params = grid_search.best_params_
print("Meilleurs paramètres après optimisation :", best_params)

# Entraîner le modèle optimisé
rf_optimized = RandomForestClassifier(random_state=42, **best_params)
rf_optimized.fit(X_train, y_train)
y_pred_optimized = rf_optimized.predict(X_test)
precision_optimized = custom_precision(y_test, y_pred_optimized)
print(f"Précision après optimisation (modèle optimisé) : {precision_optimized:.4f}")

# Visualisation
precisions = [precision_initial, precision_optimized]
labels = ['Par défaut', 'Après optimisation']

plt.figure(figsize=(15, 5))

# Barplot pour la précision
plt.subplot(1, 3, 1)
plt.bar(labels, precisions, color=['blue', 'orange'])
plt.ylabel('Précision')
plt.title('Précision avant et après optimisation')

# Visualisation du nombre de combinaisons
plt.subplot(1, 3, 2)
plt.bar(['Combinaisons évaluées'], [len(grid_search.cv_results_['params'])], color='purple')
plt.ylabel('Nombre de combinaisons')
plt.title('Combinaisons évaluées par GridSearchCV')

# Visualisation des scores moyens par combinaison
cv_results = grid_search.cv_results_
mean_test_scores = cv_results['mean_test_score']
param_combinations = range(1, len(mean_test_scores) + 1)

plt.subplot(1, 3, 3)
plt.plot(param_combinations, mean_test_scores, marker='o', color='red')
plt.xlabel('Combinaison #')
plt.ylabel('Score moyen')
plt.title('Score par combinaison')

plt.tight_layout()
plt.show()