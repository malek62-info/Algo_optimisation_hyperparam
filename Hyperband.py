import pandas as pd  # Bibliothèque pour la manipulation des données
import numpy as np  # Bibliothèque pour les calculs numériques
from sklearn.ensemble import RandomForestClassifier  # Modèle de classification basé sur les forêts aléatoires
from sklearn.model_selection import train_test_split, cross_val_score  # Ajout de la validation croisée
from sklearn.metrics import recall_score  # Fonction pour calculer le rappel (recall)
from time import time  # Module pour mesurer le temps d'exécution
from math import log, ceil  # Fonctions mathématiques utiles
from random import random  # Générateur de nombres aléatoires
import time
from sklearn.metrics import make_scorer

# Adapter custom_precision pour fonctionner avec scikit-learn
def custom_precision_scorer(estimator, X, y):
    """
      Cette fonction calcule la précision moyenne du modèle en utilisant le rappel (recall) pour les deux classes.
      Le rappel est une métrique qui mesure la capacité du modèle à identifier correctement les instances positives.
      Ici, nous calculons le rappel pour la classe 1 (pos_label=1) et la classe 0 (pos_label=0), puis nous prenons la moyenne.

      Exemple :
      Si le modèle a un rappel de 0.8 pour la classe 1 et de 0.9 pour la classe 0, la précision moyenne sera de (0.8 + 0.9) / 2 = 0.85.
      """
    y_pred = estimator.predict(X)
    recall_class_1 = recall_score(y, y_pred, pos_label=1)
    recall_class_0 = recall_score(y, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Chargement des données depuis un fichier Excel
file_path = 'C:/irace_random_forest/data_cleaned.xlsx'
df = pd.read_excel(file_path)

# Séparation des caractéristiques (X) et de la cible (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Division des données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement initial d'un modèle de forêt aléatoire avec les paramètres par défaut
rf_initial = RandomForestClassifier(random_state=42)
rf_initial.fit(X_train, y_train)
y_pred_initial = rf_initial.predict(X_test)
precision_initial = custom_precision_scorer(rf_initial, X_test, y_test)
print(f"Précision avant optimisation (modèle initial) : {precision_initial:.4f}")


class Hyperband:
    """
    Hyperband est un algorithme d'optimisation des hyperparamètres qui combine une recherche aléatoire avec une allocation de ressources dynamique.
    L'objectif est de trouver les meilleurs hyperparamètres pour un modèle en explorant efficacement l'espace des hyperparamètres.
    """

    def __init__(self, get_params_function, try_params_function, max_time=7200):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.max_iter = 200
        self.eta = 2
        self.s_max = int(log(self.max_iter) / log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter
        self.results = []
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        self.best_precision = 0  # Nouvel attribut pour stocker la meilleure précision
        self.max_time = max_time

    def run(self, skip_last=0, dry_run=False):
        start_time = time.time()
        for s in reversed(range(self.s_max + 1)):
            if time.time() - start_time > self.max_time:
                print("Temps maximum atteint, arrêt de l'exécution d'Hyperband.")
                break
            print(f"Cycle {s + 1}/{self.s_max + 1} commencé...")
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            T = [self.get_params() for _ in range(n)]
            print(f"  Nombre d'essais à évaluer dans ce cycle : {n}")
            print(f"  Nombre d'itérations par essai : {r}")
            for i in range(s + 1):
                if time.time() - start_time > self.max_time:
                    print("Temps maximum atteint, arrêt de l'exécution d'Hyperband.")
                    break
                ni = int(n * self.eta ** (-i))
                ri = int(r * self.eta ** i)
                print(f"  Essai {i + 1}/{s + 1}: Allocation de {ni} essais avec {ri} itérations chacun.")
                val_losses = []
                val_precisions = []
                for t in T:
                    self.counter += 1
                    print(f"    Test de l'hyperparamètre : {t}")
                    if dry_run:
                        result = {'loss': random(), 'precision': 1 - random()}
                    else:
                        result = self.try_params(ri, t)
                    loss = result['loss']
                    precision = result['precision']
                    val_losses.append(loss)
                    val_precisions.append(precision)
                    # Mettre à jour la meilleure perte et précision
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_precision = precision  # Mettre à jour la meilleure précision
                        self.best_counter = self.counter
                # Sélectionner les meilleures configurations
                indices = np.argsort(val_losses)[:max(1, int(ni / self.eta))]
                T = [T[i] for i in indices]
            print(f"Cycle {s + 1}/{self.s_max + 1} terminé.")
        return self.results


# Définition de l'espace de recherche des hyperparamètres
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'max_depth': [3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50],
    'min_samples_leaf': [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50],
    'max_features': ["sqrt", "log2", None],
    'criterion': ["gini", "entropy"]
}

# Fonction pour générer aléatoirement des hyperparamètres
def get_params():
    """
    Cette fonction génère aléatoirement une combinaison d'hyperparamètres à partir de l'espace de recherche défini.

    Exemple :
    Elle pourrait retourner une combinaison comme {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, ...}.
    """
    return {key: np.random.choice(values) for key, values in param_grid.items()}

# Modifier try_params pour utiliser custom_scorer
def try_params(n_iterations, params):
    model = RandomForestClassifier(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_precision_scorer)
    precision = np.mean(scores)
    return {'loss': 1 - precision, 'precision': precision}


# Exécution de l'optimisation avec Hyperband
hyperband = Hyperband(get_params, try_params)
print("\nLancement de Hyperband pour l'optimisation...")
start_time = time.time()
results = hyperband.run(skip_last=0, dry_run=False)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60

# Afficher le temps d'exécution et les meilleurs résultats
print(f"Temps d'exécution d'Hyperband : {execution_time_minutes:.2f} minutes")
if hyperband.best_params:
    print(f"Meilleure précision trouvée : {hyperband.best_precision:.4f}")
    print(f"Meilleurs hyperparamètres trouvés : {hyperband.best_params}")
else:
    print("Aucune précision n'a été calculée.")