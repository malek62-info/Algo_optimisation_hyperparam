import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import time
import csv

# Charger les données
fichier_donnees = Path("C:/tuning/data_cleaned.xlsx")
donnees = pd.read_excel(fichier_donnees)

# Utiliser les 500 premières lignes pour les tests
X = donnees.iloc[:500, :-1]
y = donnees.iloc[:500, -1]

# Séparation des données
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    try:
        recall_class_1 = recall_score(y_true, y_pred, pos_label=1, average='binary')
        recall_class_0 = recall_score(y_true, y_pred, pos_label=0, average='binary')
        return (recall_class_1 + recall_class_0) / 2
    except ValueError:
        return 0.0

# Fonction pour sauvegarder les résultats dans un fichier CSV
def save_results_csv(config, score, execution_time, filename="results_irace.csv"):
    file_exists = Path(filename).exists()
    with open(filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "criterion", "seed", "accuracy", "execution_time"])
        if not file_exists:
            writer.writeheader()  # Écrire l'entête si le fichier est nouveau
        config["accuracy"] = score
        config["execution_time"] = execution_time
        writer.writerow(config)

# Fonction d'évaluation pour irace
def evaluate_hyperparameters():
    if len(sys.argv) < 9:
        print("Erreur : Nombre insuffisant d'arguments passés à evaluate_hyperparameters.")
        sys.exit(1)

    # Lecture des hyperparamètres depuis les arguments
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    min_samples_split = int(sys.argv[3])
    min_samples_leaf = int(sys.argv[4])
    max_features = sys.argv[5]
    criterion = sys.argv[6]
    seed = int(sys.argv[7])
    instance_arg = sys.argv[8]

    if max_features == "None":
        max_features = None

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=seed
    )

    start_time = time.time()
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=custom_precision)
    score = np.mean(cv_scores)
    execution_time = time.time() - start_time

    config = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "criterion": criterion,
        "seed": seed
    }
    save_results_csv(config, score, execution_time)

    if execution_time > 7200:
        print("Temps d'exécution dépassé 2 heures.")
        sys.exit(1)

    print(score, execution_time)

if __name__ == "__main__":
    evaluate_hyperparameters()
