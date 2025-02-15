    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score
    from pathlib import Path

    # Chargement des données depuis un fichier Excel
    fichier_donnees = Path("C:/irace algo genitic/data_cleaned.xlsx")
    donnees = pd.read_excel(fichier_donnees)

    # Séparation des données en caractéristiques (X) et cible (y)
    X = donnees.iloc[:, :-1]
    y = donnees.iloc[:, -1]

    # Tester les hyperparamètres optimaux
    optimal_params = {
        'n_estimators': 173,  # Exemple d'hyperparamètres optimaux obtenus
        'max_depth': 3,
        'min_samples_split': 6,
        'min_samples_leaf': 28,
        'max_features': 'sqrt',
        'criterion': 'entropy'
    }
    def custom_precision(y_true, y_pred):
        recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
        recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
        return (recall_class_1 + recall_class_0) / 2

    # Créer un modèle RandomForestClassifier avec les hyperparamètres optimaux
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=28,
        max_depth=17,
        min_samples_split=4,
        min_samples_leaf=17,
        max_features=None,
        criterion='entropy',
        random_state=42
    )

    # Validation croisée pour évaluer les performances du modèle
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Calcul de la métrique customisée
    precision = custom_precision(y, y_pred)
    print("\n### Résultat avec les hyperparamètres optimaux ###")
    print("Custom Precision :", precision)
