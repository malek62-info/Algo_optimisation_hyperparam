import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, make_scorer
from time import time, ctime
from math import log, ceil
from random import random
import matplotlib.pyplot as plt

# Fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)  # Recall de la classe 1
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)  # Recall de la classe 0
    average_recall = (recall_class_1 + recall_class_0) / 2  # Moyenne des deux recalls
    return average_recall

# Charger les données
file_path = 'C:/irace_random_forest/data_cleaned.xlsx'  # Chemin vers vos données
df = pd.read_excel(file_path)

# Sélection des variables et de la cible
X = df.drop(columns=['Survived'])  # Variables explicatives
y = df['Survived']  # Variable cible

# Division des données en ensembles d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement initial pour comparaison (avant optimisation)
rf_initial = RandomForestClassifier(random_state=42)
rf_initial.fit(X_train, y_train)
y_pred_initial = rf_initial.predict(X_test)
precision_initial = custom_precision(y_test, y_pred_initial)
print(f"Précision avant optimisation (modèle initial) : {precision_initial:.4f}")


# Classe Hyperband pour optimiser les hyperparamètres
class Hyperband:
    def __init__(self, get_params_function, try_params_function):
        """
        Initialisation de la classe Hyperband avec la fonction de génération d'hyperparamètres et la fonction d'évaluation des configurations.
        """
        self.get_params = get_params_function  # Fonction pour générer les paramètres
        self.try_params = try_params_function  # Fonction pour essayer une configuration
        self.max_iter = 90  # Nombre maximal d'itérations par configuration
        self.eta = 2  # Taux de sous-échantillonnage (plus il est élevé, plus la réduction des configurations est importante)
        self.logeta = lambda x: log(x) / log(self.eta)  # Logarithme en base eta
        self.s_max = int(self.logeta(self.max_iter))  # Calcul du nombre maximal de stades
        self.B = (self.s_max + 1) * self.max_iter  # Budget total des ressources
        self.results = []  # Liste des résultats obtenus
        self.counter = 0  # Compteur des évaluations
        self.best_loss = np.inf  # Meilleure perte (initialisée à l'infini)
        self.best_counter = -1  # Compteur associé à la meilleure perte

    def run(self, skip_last=0, dry_run=False):
        """
        Méthode principale pour lancer l'algorithme Hyperband. Elle effectue une recherche de configurations par étapes successives.
        """
        for s in reversed(range(self.s_max + 1)):  # On parcourt les stades de l'algorithme
            # Initialisation du nombre de configurations et du nombre d'itérations par configuration
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)

            # Générer des configurations aléatoires
            T = [self.get_params() for _ in range(n)]  # Liste des configurations

            for i in range((s + 1) - int(skip_last)):  # On évite le dernier stade si "skip_last" est activé
                n_configs = n * self.eta ** (-i)  # Nombre de configurations à chaque étape
                n_iterations = r * self.eta ** (i)  # Nombre d'itérations pour chaque configuration

                print(f"\n*** {n_configs:.0f} configurations x {n_iterations:.1f} iterations each")

                val_losses = []  # Liste des pertes de validation
                early_stops = []  # Liste des arrêts anticipés

                for t in T:  # Pour chaque configuration
                    self.counter += 1
                    print(f"\n{self.counter} | {ctime()} | lowest loss so far: {self.best_loss:.4f} (run {self.best_counter})")

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params(n_iterations, t)  # Evaluer la configuration

                    assert isinstance(result, dict)  # Vérification que le résultat est un dictionnaire
                    assert 'loss' in result  # Vérification que 'loss' est présent

                    seconds = int(time() - start_time)  # Temps d'exécution de l'évaluation
                    print(f"{seconds} seconds.")

                    loss = result['loss']
                    val_losses.append(loss)

                    # Suivi de la meilleure perte
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    self.results.append(result)

                # Sélectionner les meilleures configurations pour la prochaine itération
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices]  # Trier les configurations par perte
                T = T[0:int(n_configs / self.eta)]  # Garder les meilleures configurations

        return self.results


# Fonction pour générer les hyperparamètres
def get_params():
    """
    Fonction pour générer un jeu d'hyperparamètres aléatoires pour le modèle RandomForest.
    """
    # Élargissement des plages de recherche pour les hyperparamètres
    # Élargissement des plages de recherche pour les hyperparamètres
    n_estimators = np.random.choice([50, 100, 150, 200, 300, 400, 500])
    max_depth = np.random.choice([None, 10, 20, 30, 40, 50, 60])
    min_samples_split = np.random.choice([2, 5, 10, 15, 20])
    min_samples_leaf = np.random.choice([1, 2, 4, 6, 8, 10])
    max_features = np.random.choice(['sqrt', 'log2', None])  # Suppression de 'auto'
    bootstrap = np.random.choice([True, False])

    # Retour des paramètres sous forme de dictionnaire
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap
    }


# Fonction pour essayer les hyperparamètres
def try_params(n_iterations, params):
    """
    Fonction pour entraîner un modèle avec un ensemble d'hyperparamètres et évaluer sa performance.
    """
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = custom_precision(y_test, y_pred)
    return {'loss': 1 - precision}  # On minimise la perte, donc on utilise 1 - précision


# Initialisation de Hyperband
hyperband = Hyperband(get_params, try_params)

# Exécution de Hyperband
print("\nLancement de Hyperband pour l'optimisation...")
start_time = time()
results = hyperband.run(skip_last=0, dry_run=False)
end_time = time()

# Temps d'exécution d'Hyperband
execution_time_minutes = (end_time - start_time) / 60
print(f"Temps d'exécution d'Hyperband : {execution_time_minutes:.2f} minutes")

# Affichage des meilleurs paramètres trouvés par Hyperband
best_hyperband_result = min(hyperband.results, key=lambda x: x['loss'])
best_params_hyperband = best_hyperband_result['params']
print(f"\nMeilleurs paramètres trouvés par Hyperband : {best_params_hyperband}")

# Entraînement avec les meilleurs paramètres trouvés
rf_optimized = RandomForestClassifier(random_state=42, **best_params_hyperband)
rf_optimized.fit(X_train, y_train)
y_pred_optimized = rf_optimized.predict(X_test)
precision_optimized = custom_precision(y_test, y_pred_optimized)
print(f"Précision après optimisation (modèle optimisé) : {precision_optimized:.4f}")

# Comparaison avant et après optimisation
print("\nComparaison des performances avant et après optimisation :")
print(f"Précision avant optimisation : {precision_initial:.4f}")
print(f"Précision après optimisation : {precision_optimized:.4f}")

# Visualisation des résultats avant et après optimisation
precisions = [precision_initial, precision_optimized]
labels = ['Avant optimisation', 'Après optimisation']

plt.figure(figsize=(10, 6))

# Graphique pour la précision avant et après optimisation
plt.bar(labels, precisions, color=['blue', 'green'])
plt.ylabel('Précision')
plt.title('Comparaison des précisions avant et après optimisation')

plt.tight_layout()
plt.show()
