import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import  time


# Chargement des données depuis un fichier Excel
fichier_donnees = "C:/irace algo genitic/data_cleaned.xlsx"
donnees = pd.read_excel(fichier_donnees)

# Objectif : Séparer les caractéristiques (X) et la cible (y)
# Les caractéristiques sont toutes les colonnes sauf la dernière,
# et la cible est la dernière colonne.
# Sélection des 200 premières lignes pour le test
donnees_subset = donnees.iloc[:200, :]

# Séparation des caractéristiques et de la cible sur cet échantillon
X = donnees_subset.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
y = donnees_subset.iloc[:, -1]   # Dernière colonne


# Objectif : Diviser les données en ensembles d'entraînement et de test
# Cela permet d'évaluer la performance du modèle sur des données non vues.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objectif : Définir les plages possibles pour chaque hyperparamètre
# Ces plages limitent l'espace de recherche et guident l'algorithme génétique.
n_estimators_min, n_estimators_max = 10, 500       # Nombre d'arbres dans la forêt
max_depth_min, max_depth_max = 3, 200             # Profondeur maximale des arbres
min_samples_split_min, min_samples_split_max = 2, 50  # Nombre minimal d'échantillons pour diviser un nœud
min_samples_leaf_min, min_samples_leaf_max = 1, 50    # Nombre minimal d'échantillons pour chaque feuille
max_features_categories = ["sqrt", "log2", None]      # Méthode pour sélectionner les caractéristiques
criterion_categories = ["gini", "entropy"]            # Critère de division

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def int_to_bin(value, num_bits):
    """
    Objectif : Convertir un entier en une représentation binaire avec un nombre fixe de bits.
    Exemple : int_to_bin(5, 4) -> [0, 1, 0, 1]
    """
    return np.array([int(x) for x in format(value, f'0{num_bits}b')])

def bin_to_int(binary_vector):
    """
    Objectif : Convertir un vecteur binaire en entier.
    Exemple : bin_to_int([0, 1, 0, 1]) -> 5
    """
    return int(''.join(map(str, binary_vector)), 2)

def decode_continuous(binary_vector, min_val, max_val, precision=2):
    """
    Encodage des Hyperparamètres Continus :
    Les hyperparamètres continus tels que n_estimators (nombre d'arbres), max_depth (profondeur maximale des arbres), min_samples_split et min_samples_leaf sont encodés sous forme de vecteurs binaires pour être manipulés par l'algorithme génétique.
    Décodage des Individus :
    Lorsque l'algorithme génétique évalue un individu (un ensemble d'hyperparamètres codés en binaire), la méthode decode_continuous est utilisée pour extraire les valeurs réelles des hyperparamètres continus.
    Compatibilité avec les Contraintes :
    En utilisant la formule de normalisation, cette méthode garantit que les valeurs décodées respectent toujours les plages spécifiées (min_val et max_val), ce qui évite toute valeur invalide.

    Objectif : Décoder une valeur continue à partir d'un vecteur binaire.
    Exemple : decode_continuous([1, 0, 1], 10, 50) -> ~33.33
    """
    int_val = bin_to_int(binary_vector)
    range_val = max_val - min_val
    return round(min_val + (int_val / (2**len(binary_vector) - 1)) * range_val, precision)

def decode_categorical(binary_vector, categories):
    """
    Objectif : Décoder une catégorie à partir d'un vecteur binaire.
    Exemple : decode_categorical([0, 1], ["cat1", "cat2", "cat3"]) -> "cat2"
    """
    index = bin_to_int(binary_vector)
    return categories[min(index, len(categories) - 1)]

def custom_precision(y_true, y_pred):
    """
    Objectif : Calculer une métrique personnalisée basée sur la moyenne des rappels pour les deux classes.
    Exemple :
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        custom_precision(y_true, y_pred) -> Moyenne des rappels
    """
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

def decode_individual(individual):
    """
    Objectif : Décoder un individu (vecteur binaire) en un dictionnaire d'hyperparamètres.
    Exemple :
        individual = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        decode_individual(individual) -> {"n_estimators": 42, "max_depth": 15, ...}
    """

    # Définition du nombre de bits alloués à chaque hyperparamètre
    n_estimators_bits = 8  # Nombre de bits pour n_estimators
    max_depth_bits = 4     # Nombre de bits pour max_depth
    min_samples_split_bits = 5  # Nombre de bits pour min_samples_split
    min_samples_leaf_bits = 5   # Nombre de bits pour min_samples_leaf
    max_features_bits = 2       # Nombre de bits pour max_features
    criterion_bits = 2          # Nombre de bits pour criterion

    # Extraction des segments binaires correspondant à chaque hyperparamètre
    # Les indices utilisés pour découper le vecteur sont basés sur le nombre de bits alloués
    #exemple extraction :
    #n_estimators_bin = [1, 0, 1, 0, 1, 0, 1, 0]
    #max_depth_bin = [1, 1, 0, 1]
    #min_samples_split_bin = [0, 1, 0, 1, 0]
    #min_samples_leaf_bin = [0, 0, 1, 0, 1]
    #max_features_bin = [0, 1]
    #criterion_bin = [1, 0]

    n_estimators_bin = individual[:n_estimators_bits]  # Les 8 premiers bits représentent n_estimators
    max_depth_bin = individual[n_estimators_bits:n_estimators_bits + max_depth_bits]  # Les 4 bits suivants représentent max_depth
    min_samples_split_bin = individual[n_estimators_bits + max_depth_bits:
                                       n_estimators_bits + max_depth_bits + min_samples_split_bits]  # Les 5 bits suivants représentent min_samples_split
    min_samples_leaf_bin = individual[n_estimators_bits + max_depth_bits + min_samples_split_bits:
                                      n_estimators_bits + max_depth_bits + min_samples_split_bits + min_samples_leaf_bits]  # Les 5 bits suivants représentent min_samples_leaf
    max_features_bin = individual[-(max_features_bits + criterion_bits):-criterion_bits]  # Les 2 bits avant les derniers représentent max_features
    criterion_bin = individual[-criterion_bits:]  # Les 2 derniers bits représentent criterion

    # Conversion des segments binaires en valeurs réelles ou catégorielles
    # Pour les hyperparamètres continus, on utilise decode_continuous
    #exemple de conversion
    #n_estimators = decode_continuous([1, 0, 1, 0, 1, 0, 1, 0], 10, 500)  # Résultat : ~170
    #max_depth = decode_continuous([1, 1, 0, 1], 3, 200)  # Résultat : ~135
    #min_samples_split = decode_continuous([0, 1, 0, 1, 0], 2, 50)  # Résultat : ~17
    #min_samples_leaf = decode_continuous([0, 0, 1, 0, 1], 1, 50)  # Résultat : ~10
    #max_features = decode_categorical([0, 1], ["sqrt", "log2", None])  # Résultat : "log2"
    #criterion = decode_categorical([1, 0], ["gini", "entropy"])  # Résultat : "entropy"

    n_estimators = decode_continuous(n_estimators_bin, n_estimators_min, n_estimators_max)  # Convertit n_estimators_bin en une valeur entre n_estimators_min et n_estimators_max
    max_depth = decode_continuous(max_depth_bin, max_depth_min, max_depth_max)  # Convertit max_depth_bin en une valeur entre max_depth_min et max_depth_max
    min_samples_split = decode_continuous(min_samples_split_bin, min_samples_split_min, min_samples_split_max)  # Convertit min_samples_split_bin en une valeur entre min_samples_split_min et min_samples_split_max
    min_samples_leaf = decode_continuous(min_samples_leaf_bin, min_samples_leaf_min, min_samples_leaf_max)  # Convertit min_samples_leaf_bin en une valeur entre min_samples_leaf_min et min_samples_leaf_max

    # Pour les hyperparamètres catégoriques, on utilise decode_categorical
    max_features = decode_categorical(max_features_bin, max_features_categories)  # Convertit max_features_bin en une catégorie parmi max_features_categories
    criterion = decode_categorical(criterion_bin, criterion_categories)  # Convertit criterion_bin en une catégorie parmi criterion_categories

    # Retourne un dictionnaire contenant tous les hyperparamètres décodés
    return {
        "n_estimators": int(n_estimators),  # Convertit n_estimators en entier
        "max_depth": int(max_depth),  # Convertit max_depth en entier
        "min_samples_split": int(min_samples_split),  # Convertit min_samples_split en entier
        "min_samples_leaf": int(min_samples_leaf),  # Convertit min_samples_leaf en entier
        "max_features": max_features,  # Utilise la catégorie décodée pour max_features
        "criterion": criterion  # Utilise la catégorie décodée pour criterion
    }


def fitness(individual, X, y, cv):
    """Évalue un individu avec une validation croisée 5-fold"""
    params = decode_individual(individual)

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion=params["criterion"],
        random_state=42
    )

    # Prédictions via validation croisée
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    # Calcul du score moyen
    score = custom_precision(y, y_pred)

    return score

# Paramètres de l'algorithme génétique
population_size = 100  # Taille de la population
num_generations = 50   # Nombre de générations
mutation_rate = 0.3    # Probabilité de mutation

def uniform_crossover(p1, p2):
    mask = np.random.randint(2, size=len(p1))
    child = np.where(mask, p1, p2)
    return child

def adaptive_mutation(individual, fitness_score, base_mutation_rate):
    mutation_rate = base_mutation_rate * (1 - fitness_score)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def run_genetic_algorithm(population_size, mutation_rate, X_train, X_test, y_train, y_test):
    """
    Objectif : Optimiser les hyperparamètres du modèle RandomForest à l'aide d'un algorithme génétique
    pendant 2 heures.
    """
    # Temps de fin après 2 heures
    start_time = time.time()
    end_time = start_time + 2 * 3600

    # Définition du nombre de bits alloués à chaque hyperparamètre
    n_estimators_bits = 8
    max_depth_bits = 4
    min_samples_split_bits = 5
    min_samples_leaf_bits = 5
    max_features_bits = 2
    criterion_bits = 2

    # Calcul du nombre total de bits par individu
    num_features = (n_estimators_bits + max_depth_bits + min_samples_split_bits +
                    min_samples_leaf_bits + max_features_bits + criterion_bits)

    # Initialisation de la population
    population = np.random.randint(2, size=(population_size, num_features))
    best_score = 0
    best_individual = None
    generation = 0

    while time.time() < end_time:
        generation += 1
        fitness_scores = np.array([fitness(ind, X_train, y_train, cv) for ind in population])
        current_best_score = max(fitness_scores)
        current_best_individual = population[np.argmax(fitness_scores)]

        print(f"Génération {generation} - Meilleur score : {current_best_score}")
        print("Hyperparamètres :", decode_individual(current_best_individual))

        if current_best_score > best_score:
            best_score = current_best_score
            best_individual = current_best_individuals

        parents_indices = np.random.choice(
            np.arange(population_size),
            size=population_size,
            p=fitness_scores / fitness_scores.sum() if fitness_scores.sum() != 0 else None
        )
        parents = population[parents_indices]

        children = []
        for i in range(population_size):
            p1 = parents[i]
            p2 = parents[np.random.randint(population_size)]
            # Utilisation du croisement uniforme
            child = uniform_crossover(p1, p2)
            children.append(child)

        for i in range(population_size):
            # Utilisation de la mutation adaptative
            children[i] = adaptive_mutation(children[i], fitness_scores[i], mutation_rate)

        population = np.array(children)

    print("\nMeilleur score final :", best_score)
    print("Hyperparamètres optimaux :", decode_individual(best_individual))
    return best_score


# Exécution de l'algorithme pendant 2 heures
best_score = run_genetic_algorithm(population_size, mutation_rate, X_train, X_test, y_train, y_test)
