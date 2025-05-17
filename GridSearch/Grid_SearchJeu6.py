import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, make_scorer
import time
from itertools import product

# Définition de la métrique personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Chemins des fichiers
data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/data.csv"
result_file = "GridSearchRes6.txt"

# Charger les données d'entraînement (CSV)
df_train = pd.read_csv(data_train_path,sep=',')
# Supprimer la colonne 'id' si elle existe


# Suppression de la colonne 'id'
df_train = df_train.drop(columns=['id'])
# Transformation de la variable cible : M → 1, B → 0
df_train['diagnosis'] = df_train['diagnosis'].map({'M': 1, 'B': 0})


# Sélection des variables et de la cible
X_train = df_train.drop(columns=['diagnosis'])
y_train = df_train['diagnosis']

# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [20, 30, 50, 75, 100, 150, 200],
    'max_depth': [5, 10, 20, 25, 30],
    'min_samples_split': [2, 3, 5, 7, 10, 12],
    'min_samples_leaf': [1, 2, 3, 7, 5, 10, 12],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'ccp_alpha': [0.003]
}

# Initialisation
best_score = -float('inf')
best_params = None
start_time = time.time()
time_limit = 7200  # 2 heures en secondes
custom_scorer = make_scorer(custom_precision)
results = {'mean_test_score': []}

# Générer toutes les combinaisons possibles d'hyperparamètres
param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

print("Début de la recherche manuelle des hyperparamètres...")

# Boucle sur toutes les combinaisons
for i, combination in enumerate(param_combinations, 1):
    params = dict(zip(param_keys, combination))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=custom_scorer, n_jobs=-1)
    mean_score = np.mean(scores)

    results['mean_test_score'].append(mean_score)
    elapsed_time = time.time() - start_time

    if mean_score > best_score and elapsed_time < time_limit:
        best_score = mean_score
        best_params = params
        with open(result_file, 'w') as f:
            f.write(f"Best Score: {best_score:.4f}\n")
            for key in best_params:
                f.write(f"{key}: {best_params[key]}\n")
        print(f"Itération {i} : Nouveau meilleur score {best_score:.4f}, écrit dans {result_file}")

    print(f"Itération {i}/{len(param_combinations)} - Score : {mean_score:.4f}, Temps écoulé : {elapsed_time / 60:.2f} min")

    if elapsed_time >= time_limit:
        print("Limite de temps dépassée, arrêt de la recherche.")
        break

total_elapsed_time = time.time() - start_time
print(f"Temps total d'exécution de la recherche : {total_elapsed_time / 60:.2f} minutes")
