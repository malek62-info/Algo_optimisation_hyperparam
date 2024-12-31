import sys
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

# Fonction d'évaluation pour irace
def evaluate_hyperparameters():
    # Vérifiez le nombre d'arguments
    if len(sys.argv) < 7:
        print("Erreur : Nombre insuffisant d'arguments passés à evaluate_hyperparameters.")
        sys.exit(1)

    # Lecture des hyperparamètres depuis les arguments
    ntree = int(sys.argv[1])  # Nombre d'arbres (ntree)
    mtry = int(sys.argv[2])   # Nombre de variables à tester pour chaque split (mtry)
    seed = int(sys.argv[3])   # Graine pour la reproductibilité
    instance_arg = sys.argv[4]  # Instance (pas utilisé ici, mais souvent fourni par irace)
    
    # Créer et entraîner le modèle RandomForest
    clf = RandomForestClassifier(n_estimators=ntree, max_features=mtry, random_state=seed)
    
    # Entraîner le modèle sur les données d'entraînement
    clf.fit(X_train, y_train)
    
    # Faire des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)
    
    # Calculer la précision personnalisée
    score = custom_precision(y_test, y_pred)
    
    # Afficher le score et le temps d'exécution (simulé ici)
    import time
    start_time = time.time()
    execution_time = time.time() - start_time
    
    # Afficher les résultats dans le format attendu par irace
    print(score, execution_time)

if __name__ == "__main__":
    evaluate_hyperparameters()
