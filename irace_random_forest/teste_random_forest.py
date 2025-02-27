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
def test_best_random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion,
                            random_state):
    """
    Teste un modèle Random Forest avec les meilleurs hyperparamètres trouvés par irace.

    :param n_estimators: Le nombre d'arbres (n_estimators)
    :param max_depth: La profondeur maximale de l'arbre (max_depth)
    :param min_samples_split: Le nombre minimal d'échantillons pour diviser un nœud (min_samples_split)
    :param min_samples_leaf: Le nombre minimal d'échantillons dans une feuille (min_samples_leaf)
    :param max_features: Le nombre de caractéristiques à considérer pour chaque split (max_features)
    :param criterion: Le critère de division ("gini" ou "entropy")
    :param random_state: La graine pour la reproductibilité
    """
    # Gestion de la valeur None pour max_features
    if max_features == "None":
        max_features = None

    # Créer le modèle RandomForest avec les meilleurs hyperparamètres
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=random_state
    )

    # Entraîner le modèle sur les données d'entraînement
    clf.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Calculer la précision personnalisée
    score = custom_precision(y_test, y_pred)

    # Afficher la précision personnalisée
    print(f"Précision personnalisée (moyenne des rappels pour les classes 0 et 1): {score:.4f}")


# Exemple d'utilisation avec les meilleurs hyperparamètres trouvés
# Ces hyperparamètres peuvent provenir de la sortie de irace
best_n_estimators = 444  # Nombre d'arbres
best_max_depth = 150  # Profondeur maximale de l'arbre
best_min_samples_split = 44  # Nombre minimal d'échantillons pour diviser un nœud
best_min_samples_leaf = 12  # Nombre minimal d'échantillons dans une feuille
best_max_features = "None"  # Nombre de caractéristiques à considérer pour chaque split
best_criterion = "gini"  # Critère de division
best_random_state = 42  # Graine pour la reproductibilité

# Test du modèle avec les meilleurs hyperparamètres
test_best_random_forest(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    max_features=best_max_features,
    criterion=best_criterion,
    random_state=best_random_state
)