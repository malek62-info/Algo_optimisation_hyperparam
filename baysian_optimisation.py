import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import time

# Définir votre fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    average_recall = (recall_class_1 + recall_class_0) / 2
    return average_recall

# Charger les données depuis un fichier Excel
df = pd.read_excel('C:/irace_random_forest/data_cleaned.xlsx')
print(df.head())

# Séparer les features et la cible
X = df.drop(columns=['Survived'])
y = df['Survived']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du RandomForest avec des paramètres par défaut
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Afficher l'importance des variables
importances = rf_default.feature_importances_
indices = np.argsort(importances)[::-1]
selected_features = X.columns[indices[:10]]
print(f"Les 10 variables les plus importantes sont : {selected_features}")
print(f"Importance des variables : {importances[indices[:10]]}")

# Prédiction avec le modèle par défaut
y_pred_default = rf_default.predict(X_test)
precision_default = custom_precision(y_test, y_pred_default)
print(f"Précision avant optimisation (modèle par défaut) : {precision_default:.4f}")

# Définir l'espace des hyperparamètres pour l'optimisation bayésienne
param_dist = {
    'n_estimators': (10, 500),  # Nombre d'arbres dans la forêt
    'max_depth': (3, 200),  # Profondeur maximale des arbres
    'min_samples_split': (2, 50),  # Nombre minimal d'échantillons pour diviser un nœud
    'min_samples_leaf': (1, 50),  # Nombre minimal d'échantillons pour chaque feuille
    'max_features': ["sqrt", "log2", None],  # Méthode pour sélectionner les caractéristiques
    'criterion': ["gini", "entropy"],  # Critère de division
}

# Limiter le temps d'exécution à 2 heures (7200 secondes)
start_time = time.time()
time_limit = 7200  # 2 heures en secondes

# Optimisation bayésienne avec un nombre d'itérations réduit
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_dist,
    n_iter=50,  # Nombre d'itérations pour explorer l'espace des hyperparamètres
    cv=5,  # Réduit le nombre de folds pour accélérer
    verbose=3,
    n_jobs=-1,  # Utiliser tous les cœurs disponibles
    random_state=42
)

# Entraîner le modèle avec l'optimisation bayésienne
bayes_search.fit(X_train, y_train)

# Calculer le temps écoulé pour l'optimisation bayésienne
execution_time = time.time() - start_time
print(f"Temps d'exécution de l'optimisation bayésienne : {execution_time:.4f} secondes")

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres après optimisation bayésienne : ", bayes_search.best_params_)

# Entraîner le modèle optimisé avec les meilleurs paramètres
rf_optimized = RandomForestClassifier(
    random_state=42,
    **bayes_search.best_params_
)
rf_optimized.fit(X_train, y_train)

# Afficher l'importance des variables pour le modèle optimisé
importances_best = rf_optimized.feature_importances_
indices_best = np.argsort(importances_best)[::-1]
selected_features_best = X.columns[indices_best[:10]]
print(f"Les 10 variables les plus importantes après optimisation : {selected_features_best}")
print(f"Importance des variables : {importances_best[indices_best[:10]]}")

# Prédiction avec le modèle optimisé
y_pred_best = rf_optimized.predict(X_test)
precision_best = custom_precision(y_test, y_pred_best)
print(f"Précision après optimisation (modèle optimisé) : {precision_best:.4f}")

# Visualisation des résultats
results = {
    'before_optimization': {
        'precision': precision_default,
        'selected_features': selected_features,
        'importances': importances[indices[:10]]
    },
    'after_optimization': {
        'precision': precision_best,
        'selected_features': selected_features_best,
        'importances': importances_best[indices_best[:10]]
    }
}

# Tracer la précision avant et après optimisation
precisions = [precision_default, precision_best]
labels = ['Avant optimisation', 'Après optimisation']
plt.bar(labels, precisions, color=['blue', 'green'])
plt.ylabel('Précision')
plt.title('Précision avant et après optimisation bayésienne')
plt.show()

# Tracer les résultats de l'optimisation bayésienne
results_df = pd.DataFrame(bayes_search.cv_results_)
plt.figure(figsize=(12, 8))
plt.scatter(results_df['param_n_estimators'], results_df['param_max_depth'], c=results_df['mean_test_score'], cmap='viridis')
plt.colorbar(label='Mean Test Score (Accuracy)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.title('Accuracy de chaque combinaison d\'hyperparamètres')
plt.show()

# Afficher les résultats finaux
print(results)