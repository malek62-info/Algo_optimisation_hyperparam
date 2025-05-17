import random
import numpy as np
from deap import base, creator, tools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import time

# Taux de mutation
MUTATION_PROB = 0.05

# Fichier de sauvegarde
output_file = "genetic_optimization_results.txt"

def custom_recall(true_labels, predictions):
    recall_class1 = recall_score(true_labels, predictions, pos_label=1)
    recall_class0 = recall_score(true_labels, predictions, pos_label=0)
    return (recall_class1 + recall_class0) / 2

# Chargement des données
train_data = pd.read_excel("C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_train_t3.xlsx")
X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']



# Espace des hyperparamètres
param_ranges = {
    'n_estimators': (10, 200),
    'max_depth': (5, 50),
    'min_samples_split': (2, 12),
    'min_samples_leaf': (1, 12),
    'max_features': ["sqrt", "log2"],
    'criterion': ["gini", "entropy"],
    'class_weight': ['balanced', 'balanced_subsample'],
    'ccp_alpha': [0.003, 0.004]
}

# Création des classes DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Génération des attributs
toolbox.register("attr_n_estimators", random.randint, *param_ranges['n_estimators'])
toolbox.register("attr_max_depth", random.randint, *param_ranges['max_depth'])
toolbox.register("attr_min_samples_split", random.randint, *param_ranges['min_samples_split'])
toolbox.register("attr_min_samples_leaf", random.randint, *param_ranges['min_samples_leaf'])
toolbox.register("attr_max_features", random.choice, param_ranges['max_features'])
toolbox.register("attr_criterion", random.choice, param_ranges['criterion'])
toolbox.register("attr_class_weight", random.choice, param_ranges['class_weight'])
toolbox.register("attr_ccp_alpha", random.choice, param_ranges['ccp_alpha'])

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_estimators,
                  toolbox.attr_max_depth,
                  toolbox.attr_min_samples_split,
                  toolbox.attr_min_samples_leaf,
                  toolbox.attr_max_features,
                  toolbox.attr_criterion,
                  toolbox.attr_class_weight,
                  toolbox.attr_ccp_alpha), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_individual(individual):
    model = RandomForestClassifier(
        n_estimators=individual[0],
        max_depth=individual[1],
        min_samples_split=individual[2],
        min_samples_leaf=individual[3],
        max_features=individual[4],
        criterion=individual[5],
        class_weight=individual[6],
        ccp_alpha=individual[7],
        random_state=42
    )
    scorer = make_scorer(custom_recall)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
    return np.mean(scores),

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)

def mutate_individual(individual):
    idx = random.randint(0, len(individual) - 1)
    param_name = list(param_ranges.keys())[idx]
    old_value = individual[idx]

    if idx < 4:
        individual[idx] = random.randint(*param_ranges[param_name])
        while individual[idx] == old_value:
            individual[idx] = random.randint(*param_ranges[param_name])
    elif idx == 7:
        options = param_ranges[param_name]
        if len(options) == 2:
            individual[idx] = round(options[1] if old_value == options[0] else options[0], 4)
        else:
            new_value = round(random.uniform(*options), 4)
            while new_value == old_value:
                new_value = round(random.uniform(*options), 4)
            individual[idx] = new_value
    else:
        options = param_ranges[param_name]
        if len(options) == 2:
            individual[idx] = options[1] if old_value == options[0] else options[0]
        else:
            new_value = random.choice(options)
            while new_value == old_value:
                new_value = random.choice(options)
            individual[idx] = new_value
    return individual

toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=5)

def main():
    pop_size = 50
    n_generations = 40
    time_limit = 7200

    population = toolbox.population(n=pop_size)
    best_scores = []
    best_individual = None
    best_score = -np.inf
    save_stopped_gen = None
    start_time = time.time()

    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_generations):
        print(f"\n-- Génération {gen + 1} --")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fitness_values = [ind.fitness.values[0] for ind in population]
        current_best = tools.selBest(population, k=1)[0]
        current_score = current_best.fitness.values[0]
        best_scores.append(current_score)

        elapsed_time = time.time() - start_time
        if elapsed_time <= time_limit and current_score > best_score:
            best_score = current_score
            best_individual = current_best
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Custom Recall: {best_score:.4f}\n")
                f.write("Hyperparameters:\n")
                f.write(f"n_estimators: {best_individual[0]}\n")
                f.write(f"max_depth: {best_individual[1]}\n")
                f.write(f"min_samples_split: {best_individual[2]}\n")
                f.write(f"min_samples_leaf: {best_individual[3]}\n")
                f.write(f"max_features: {best_individual[4]}\n")
                f.write(f"criterion: {best_individual[5]}\n")
                f.write(f"class_weight: {best_individual[6]}\n")
                f.write(f"ccp_alpha: {best_individual[7]}\n")
            print(f"\nNouveau meilleur score sauvegardé ! Recall : {best_score:.4f}")
        elif elapsed_time > time_limit and save_stopped_gen is None:
            save_stopped_gen = gen + 1
            print("\nLimite de temps dépassée, arrêt de la sauvegarde.")

        model = RandomForestClassifier(
            n_estimators=current_best[0],
            max_depth=current_best[1],
            min_samples_split=current_best[2],
            min_samples_leaf=current_best[3],
            max_features=current_best[4],
            criterion=current_best[5],
            class_weight=current_best[6],
            ccp_alpha=current_best[7],
            random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_train, model.predict(X_train))

        print(f" Best individual : {current_best}")
        print(f" Fitness (custom recall): {current_score:.4f}")
        print(f" Training accuracy: {accuracy:.4f}")
        print(f" Moyenne population : {sum(fitness_values)/len(population):.4f}")

    total_time = time.time() - start_time
    print(f"\nTemps total : {total_time / 60:.2f} minutes")

    best_individual_final = tools.selBest(population, k=1)[0]
    print("\nMeilleur individu final :", best_individual_final)

    if total_time <= time_limit:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\nTemps d'exécution : {total_time / 60:.2f} minutes\n")
        print("\nConfiguration finale sauvegardée.")
    else:
        print("\nTemps limite dépassé, configuration finale non sauvegardée.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_generations + 1), best_scores, marker='o', color='blue', linestyle='-')
    if save_stopped_gen is not None:
        plt.plot(range(save_stopped_gen, n_generations + 1), best_scores[save_stopped_gen - 1:], marker='o', color='red', linestyle='-')
    plt.xlabel('Génération')
    plt.ylabel('Score (Custom Recall)')
    plt.title('Évolution du score Custom Recall')
    plt.grid(True)
    plt.show()

    return best_individual_final

if __name__ == "__main__":
    best_solution = main()
    print("\nOptimisation terminée !")
