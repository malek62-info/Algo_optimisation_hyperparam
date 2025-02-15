import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, make_scorer
import time

# Fonction de précision
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Charger les données
file_path = 'C:/irace_random_forest/data_cleaned.xlsx'
df = pd.read_excel(file_path)

# Sélection des variables et de la cible
X = df.drop(columns=['Survived'])
y = df['Survived']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition de l'espace de recherche
param_dist = {
    'n_estimators': np.linspace(10, 500, 50, dtype=int).tolist(),
    'max_depth': np.linspace(3, 200, 50, dtype=int).tolist(),
    'min_samples_split': np.linspace(2, 50, 10, dtype=int).tolist(),
    'min_samples_leaf': np.linspace(1, 50, 10, dtype=int).tolist(),
    'max_features': ["sqrt", "log2", None],
    'criterion': ["gini", "entropy"]
}

# Configuration de la recherche
custom_scorer = make_scorer(custom_precision)
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=1000,  # Un grand nombre d'itérations
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# Exécuter la recherche
total_time = 2 * 3600  # 2 heures en secondes
start_time = time.time()
random_search.fit(X_train, y_train)

# Afficher uniquement l'accuracy de chaque combinaison de paramètres testée
cv_results = random_search.cv_results_
for i in range(len(cv_results['params'])):
    params = cv_results['params'][i]
    mean_accuracy = cv_results['mean_test_score'][i]
    print(f"Combinaison {i+1}: {params} -> Accuracy: {mean_accuracy:.4f}")

# Vérifier le temps écoulé
elapsed_time = time.time() - start_time
if elapsed_time > total_time:
    print("Temps écoulé : 2 heures, arrêt de la recherche.")

# Meilleurs paramètres
best_params = random_search.best_params_
print("Meilleurs paramètres après optimisation :", best_params)

# Entraîner le modèle final
rf_optimized = RandomForestClassifier(random_state=42, **best_params)
rf_optimized.fit(X_train, y_train)
y_pred_optimized = rf_optimized.predict(X_test)
precision_optimized = custom_precision(y_test, y_pred_optimized)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"Précision après optimisation : {precision_optimized:.4f}")
print(f"Accuracy après optimisation : {accuracy_optimized:.4f}")