import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Charger les données
fichier_donnees = Path("C:/tuning/data_cleaned.xlsx")
donnees = pd.read_excel(fichier_donnees)

# Utiliser les 500 premières lignes pour les tests
X = donnees.iloc[:500, :-1]
y = donnees.iloc[:500, -1]

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    try:
        # Calcul du rappel pour la classe 1 (positive) et la classe 0 (négative)
        recall_class_1 = recall_score(y_true, y_pred, pos_label=1, average='binary')
        recall_class_0 = recall_score(y_true, y_pred, pos_label=0, average='binary')
        # Retourner la moyenne des rappels
        return (recall_class_1 + recall_class_0) / 2
    except ValueError:
        return 0.0

# Fonction pour tester Random Forest avec les meilleurs hyperparamètres trouvés par irace
def test_best_random_forest(ntree, mtry, seed,sampe_size):
    """
    Teste un modèle Random Forest avec les meilleurs hyperparamètres trouvés par irace.

    :param ntree: Le nombre d'arbres (n_estimators)
    :param mtry: Le nombre de caractéristiques à tester pour chaque séparation (max_features)
    :param seed: La graine pour la reproductibilité
    """
    # Créer le modèle RandomForest avec les meilleurs hyperparamètres
    clf = RandomForestClassifier(n_estimators=ntree, max_features=mtry, random_state=seed)

    # Entraîner le modèle sur les données d'entraînement
    clf.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Calculer la précision personnalisée
    score = custom_precision(y_test, y_pred)
    
    # Afficher la précision personnalisée
    print(f"Précision personnalisée (moyenne des rappels pour les classes 0 et 1): {score:.4f}")

# Exemple d'utilisation avec les meilleurs hyperparamètres trouvés
# Ces hyperparamètres peuvent provenir de la sortie de irace (à titre d'exemple ici)
best_ntree = 448  # Remplacez ceci par le meilleur ntree trouvé par irace
best_mtry = 8     # Remplacez ceci par le meilleur mtry trouvé par irace
best_seed = 43    # Remplacez ceci par la graine choisie par irace
sampe_size =0.6142
# Test du modèle avec les meilleurs hyperparamètres
test_best_random_forest(best_ntree, best_mtry, best_seed,sampe_size)
