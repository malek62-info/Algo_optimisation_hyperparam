import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from pathlib import Path

# Chargement des données depuis un fichier Excel
fichier_donnees = Path("C:/irace algo genitic/data_cleaned.xlsx")
donnees = pd.read_excel(fichier_donnees)

# Séparation des données en caractéristiques (X) et cible (y)
X = donnees.iloc[:, :-1]
y = donnees.iloc[:, -1]

# Plages des hyperparamètres
n_estimators_min, n_estimators_max = 10, 500
max_depth_min, max_depth_max = 3, 200
min_samples_split_min, min_samples_split_max = 2, 50
min_samples_leaf_min, min_samples_leaf_max = 1, 50
max_features_categories = ["sqrt", "log2", None]
criterion_categories = ["gini", "entropy"]

# Fonctions d'encodage et de décodage
def int_to_bin(value, num_bits):
    return np.array([int(x) for x in format(value, f'0{num_bits}b')])

def bin_to_int(binary_vector):
    return int(''.join(map(str, binary_vector)), 2)

def decode_continuous(binary_vector, min_val, max_val, precision=2):
    int_val = bin_to_int(binary_vector)
    range_val = max_val - min_val
    return round(min_val + (int_val / (2**len(binary_vector) - 1)) * range_val, precision)

def decode_categorical(binary_vector, categories):
    index = bin_to_int(binary_vector)
    return categories[min(index, len(categories) - 1)]

def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

# Décodage des individus
def decode_individual(individual):
    n_estimators_bits = 8
    max_depth_bits = 4
    min_samples_split_bits = 5
    min_samples_leaf_bits = 5
    max_features_bits = 2
    criterion_bits = 2

    n_estimators_bin = individual[:n_estimators_bits]
    max_depth_bin = individual[n_estimators_bits:n_estimators_bits + max_depth_bits]
    min_samples_split_bin = individual[n_estimators_bits + max_depth_bits:
                                       n_estimators_bits + max_depth_bits + min_samples_split_bits]
    min_samples_leaf_bin = individual[n_estimators_bits + max_depth_bits + min_samples_split_bits:
                                       n_estimators_bits + max_depth_bits + min_samples_split_bits + min_samples_leaf_bits]
    max_features_bin = individual[-(max_features_bits + criterion_bits):-criterion_bits]
    criterion_bin = individual[-criterion_bits:]

    n_estimators = bin_to_int(n_estimators_bin) + n_estimators_min
    max_depth = bin_to_int(max_depth_bin) + max_depth_min
    min_samples_split = bin_to_int(min_samples_split_bin) + min_samples_split_min
    min_samples_leaf = bin_to_int(min_samples_leaf_bin) + min_samples_leaf_min
    max_features = decode_categorical(max_features_bin, max_features_categories)
    criterion = decode_categorical(criterion_bin, criterion_categories)

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "criterion": criterion
    }

# Fonction d'évaluation (fitness)
def fitness(individual, X_train, X_test, y_train, y_test):
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
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = custom_precision(y_test, y_pred)
    return score

# Algorithme génétique
def run_genetic_algorithm(population_size, num_generations, mutation_rate, X_train, X_test, y_train, y_test):
    n_estimators_bits = 8
    max_depth_bits = 4
    min_samples_split_bits = 5
    min_samples_leaf_bits = 5
    max_features_bits = 2
    criterion_bits = 2

    num_features = (n_estimators_bits + max_depth_bits + min_samples_split_bits +
                    min_samples_leaf_bits + max_features_bits + criterion_bits)

    population = np.random.randint(2, size=(population_size, num_features))
    scores_par_generation = []
    best_individual = None
    best_score = 0

    for generation in range(num_generations):
        fitness_scores = np.array([fitness(ind, X_train, X_test, y_train, y_test) for ind in population])
        current_best_score = max(fitness_scores)
        current_best_individual = population[np.argmax(fitness_scores)]

        if current_best_score > best_score:
            best_score = current_best_score
            best_individual = current_best_individual

        print(f"Génération {generation + 1}/{num_generations} - Meilleur score : {current_best_score}")
        print("Hyperparamètres :", decode_individual(current_best_individual))

        parents_indices = np.random.choice(np.arange(population_size), size=population_size,
                                           p=fitness_scores / fitness_scores.sum() if fitness_scores.sum() != 0 else None)
        parents = population[parents_indices]
        children = np.zeros((population_size, num_features), dtype=int)

        for i in range(population_size):
            p1 = parents[i]
            p2 = parents[np.random.randint(population_size)]
            crossover_points = np.sort(np.random.choice(range(1, num_features), size=2, replace=False))
            children[i][:crossover_points[0]] = p1[:crossover_points[0]]
            children[i][crossover_points[0]:crossover_points[1]] = p2[crossover_points[0]:crossover_points[1]]
            children[i][crossover_points[1]:] = p1[crossover_points[1]:]

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_features)
                children[i, mutation_point] = 1 - children[i, mutation_point]

        population = children
        scores_par_generation.append(best_score)

    print("\nMeilleur score final :", best_score)
    print("Hyperparamètres optimaux :", decode_individual(best_individual))

    return best_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

population_size = 100
num_generations = 50
mutation_rate = 0.3

best_score = run_genetic_algorithm(population_size, num_generations, mutation_rate, X_train, X_test, y_train, y_test)
