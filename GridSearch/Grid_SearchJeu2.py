import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, make_scorer
import time
from itertools import product

# Définition de la métrique personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Chemins des fichiers
data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/B_instance.xlsx"
result_file = "GridSearchRes1.txt"

# Charger les données d'entraînement
df_train = pd.read_excel(data_train_path)

# Vérifier si la variable cible est en format chaîne et la convertir
if df_train['target'].dtype == object:
    df_train['target'] = df_train['target'].map(lambda x: 1 if x == "VRAI" else 0)

# Sélection des variables et de la cible
X_train = df_train.drop(columns=['target'])
y_train = df_train['target']



# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [20,30,50,75,100,150,200],
    'max_depth': [5, 10, 20, 25,30],
    'min_samples_split': [2, 3,5,7, 10,12],
    'min_samples_leaf': [1, 2,3,7, 5,10,12],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy','log_loss'],
    'class_weight': ['balanced','balanced_subsample'],
    'ccp_alpha': [0.003]
}

# Initialisation
best_score = -float('inf')  # Meilleur score initial
best_params = None
start_time = time.time()
time_limit = 7200  # 2 minutes en secondes (ajusté pour tester rapidement)
custom_scorer = make_scorer(custom_precision)
results = {'mean_test_score': []}

# Générer toutes les combinaisons possibles d'hyperparamètres
param_combinations = list(product(*param_grid.values()))
#Récupérer les noms des hyperparamètres
param_keys = list(param_grid.keys())

print("Début de la recherche manuelle des hyperparamètres...")

# Boucle sur toutes les combinaisons
for i, combination in enumerate(param_combinations, 1):
    # Créer un dictionnaire avec les paramètres actuels
    params = dict(zip(param_keys, combination))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Créer et évaluer le modèle
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=custom_scorer, n_jobs=-1)
    mean_score = np.mean(scores)

    # Ajouter le score aux résultats pour le graphique
    results['mean_test_score'].append(mean_score)

    # Vérifier le temps écoulé
    elapsed_time = time.time() - start_time

    # Si le score est meilleur et que le temps est inférieur à la limite, mettre à jour
    if mean_score > best_score and elapsed_time < time_limit:
        best_score = mean_score
        best_params = params
        # Écriture simplifiée dans le fichier
        with open(result_file, 'w') as f:
            f.write(f"Best Score: {best_score:.4f}\n")
            f.write(f"n_estimators: {best_params['n_estimators']}\n")
            f.write(f"max_depth: {best_params['max_depth']}\n")
            f.write(f"min_samples_split: {best_params['min_samples_split']}\n")
            f.write(f"min_samples_leaf: {best_params['min_samples_leaf']}\n")
            f.write(f"max_features: {best_params['max_features']}\n")
            f.write(f"criterion: {best_params['criterion']}\n")
            f.write(f"class_weight: {best_params['class_weight']}\n")  # Ajout de class_weight
            f.write(f"ccp_alpha: {best_params['ccp_alpha']}\n")  # Ajout de class_weight

        print(f"Itération {i} : Nouveau meilleur score {best_score:.4f}, écrit dans {result_file}")

    # Afficher l'avancement
    print(f"Itération {i}/{len(param_combinations)} - Score : {mean_score:.4f}, Temps écoulé : {elapsed_time / 60:.2f} min")

    # Arrêter si la limite de temps est dépassée
    if elapsed_time >= time_limit:
        print("Limite de temps dépassée, arrêt de la recherche.")
        break

# Calculer le temps total écoulé
total_elapsed_time = time.time() - start_time
print(f"Temps total d'exécution de la recherche : {total_elapsed_time / 60:.2f} minutes")
