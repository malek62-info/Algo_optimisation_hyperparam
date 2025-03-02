import random
import numpy as np
from deap import base, creator, tools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import cross_val_score  # Ajout pour la validation croisée
from sklearn.metrics import make_scorer  # Ajout pour créer un scorer personnalisé
import matplotlib.pyplot as plt
import time


("""Quoi ? Cette fonction calcule la moyenne des rappels (recall) des classes 0 et 1. Le rappel est la proportion de vrais positifs détectés pour une classe (ex. TP / (TP + FN)).
Pourquoi ? C’est notre critère d’optimisation (la "fitness"). On veut que le modèle soit bon pour détecter à la fois les 0 et les 1, pas seulement une classe.
Exemple :
Si y_true = [0, 1, 0, 1] et y_pred = [0, 1, 1, 1] :
Rappel classe 1 : 2/2 = 1.0 (tous les 1 sont détectés).
Rappel classe 0 : 1/2 = 0.5 (un seul 0 détecté sur deux).
custom_precision = (1.0 + 0.5) / 2 = 0.75.""")
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)  # Rappel pour la classe 1
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)  # Rappel pour la classe 0
    return (recall_class_1 + recall_class_0) / 2  # Moyenne des deux


# 📌 Chargement des données prédéfinies depuis les fichiers Excel
data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_train_t3.xlsx"
data_test_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_test_t3.xlsx"
df_train = pd.read_excel(data_train_path)  # Données d’entraînement
df_test = pd.read_excel(data_test_path)  # Données de test

# 📌 Séparation des features et labels
# Exemple : X_train contient toutes les colonnes sauf "Survived", y_train contient "Survived".
X_train = df_train.drop(columns=['Survived'])  # Features d’entraînement
y_train = df_train['Survived']  # Labels d’entraînement
X_test = df_test.drop(columns=['Survived'])  # Features de test
y_test = df_test['Survived']  # Labels de test

# 📌 Définition des plages d’hyperparamètres
# Ces plages limitent les valeurs possibles pour chaque hyperparamètre.
param_ranges = {
    'n_estimators': (10, 500),  # Nombre d’arbres dans la forêt
    'max_depth': (3, 50),  # Profondeur maximale des arbres
    'min_samples_split': (2, 10),  # Minimum d’échantillons pour diviser un nœud
    'min_samples_leaf': (1, 5),  # Minimum d’échantillons dans une feuille
    'max_features': ["sqrt", "log2"],  # Méthode pour choisir le nombre de features
    'criterion': ["gini", "entropy"]  # Critère de division des nœuds
}


("""Quoi ? On définit deux classes :
FitnessMax : une classe pour représenter la fitness (le score à maximiser), ici custom_precision.
Individual : une classe pour représenter un individu, qui est une liste d’hyperparamètres avec une fitness associée.
Pourquoi ? DEAP ne fournit pas de structure prédéfinie pour les individus. creator permet de créer des conteneurs
 personnalisés. weights=(1.0,) indique qu’on maximise un seul objectif (contrairement à une optimisation multi-objectifs).
Exemple : Un individu pourrait être [150, 20, 4, 3, "sqrt", "gini"], avec une fitness comme (0.85,).""")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 📌 Initialisation de la boîte à outils (toolbox)
toolbox = base.Toolbox()

(""" Quoi ? On configure la toolbox pour générer des individus et une population :
Générateurs : Chaque attr_ génère une valeur aléatoire (ex. attr_n_estimators peut donner 200).
Individu : initCycle combine les générateurs pour créer une liste de 6 hyperparamètres.
Population : initRepeat crée une liste d’individus.
Pourquoi ? La toolbox est le cœur de l’AG dans DEAP. Elle organise les outils pour créer et manipuler la population.
Exemple :
Appel de toolbox.individual() : [150, 20, 4, 3, "sqrt", "gini"].
Appel de toolbox.population(n=3) : [[150, 20, 4, 3, "sqrt", "gini"], [200, 10, 5, 2, "log2", "entropy"], 
[100, 15, 3, 1, "sqrt", "entropy"]].""")
toolbox.register("attr_n_estimators", random.randint, *param_ranges['n_estimators'])
toolbox.register("attr_max_depth", random.randint, *param_ranges['max_depth'])
toolbox.register("attr_min_samples_split", random.randint, *param_ranges['min_samples_split'])
toolbox.register("attr_min_samples_leaf", random.randint, *param_ranges['min_samples_leaf'])
toolbox.register("attr_max_features", random.choice, param_ranges['max_features'])
toolbox.register("attr_criterion", random.choice, param_ranges['criterion'])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_estimators, toolbox.attr_max_depth, toolbox.attr_min_samples_split,
                  toolbox.attr_min_samples_leaf, toolbox.attr_max_features, toolbox.attr_criterion),
                 n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


("""Quoi ? Cette fonction entraîne un modèle avec les hyperparamètres d’un individu et calcule son score custom_precision.
Pourquoi ? C’est la "fitness" qui indique à l’AG si un individu est bon ou non. Plus le score est élevé, meilleur est l’individu.
Exemple : Pour [150, 20, 4, 3, "sqrt", "gini"], si le modèle obtient un recall de 0.9 pour la classe 1 et 0.8 pour la classe 0,
 alors score = 0.85, retourné comme (0.85,).""")
def evaluate(individual):
    model = RandomForestClassifier(
        n_estimators=individual[0],  # Ex. 150
        max_depth=individual[1],  # Ex. 20
        min_samples_split=individual[2],  # Ex. 4
        min_samples_leaf=individual[3],  # Ex. 3
        max_features=individual[4],  # Ex. "sqrt"
        criterion=individual[5],  # Ex. "gini"
        random_state=42
    )
    # Utilisation de la validation croisée à 5 plis avec custom_precision
    custom_scorer = make_scorer(custom_precision)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer)
    score = np.mean(scores)  # Moyenne des scores sur les 5 plis
    return score,  # Tuple requis par DEAP


toolbox.register("evaluate", evaluate)


("""Quoi ? On définit trois opérateurs :
Croisement (mate) : cxTwoPoint échange des segments entre deux individus.
Mutation : Modifie aléatoirement un hyperparamètre.
Sélection : selTournament choisit les meilleurs individus par tournoi.
Pourquoi ? Ces opérateurs simulent l’évolution :
Croisement mélange les "bons gènes".
Mutation introduit de la diversité.
Sélection garde les meilleurs.
Exemple :
Croisement : [150, 20, 4, 3, "sqrt", "gini"] et [200, 10, 5, 2, "log2", "entropy"] → [150, 10, 5, 2, "log2", "entropy"].
Mutation : [150, 20, 4, 3, "sqrt", "gini"] → [150, 25, 4, 3, "sqrt", "gini"].
Sélection : Parmi [0.85], [0.78], [0.90], le tournoi choisit [0.90].""")
toolbox.register("mate", tools.cxTwoPoint)
def mutate(individual):
    ("""Exemple :
individual = [150, 20, 4, 3, "sqrt", "gini"], index = 2 (min_samples_split).
param_name = "min_samples_split", plage = (2, 10).
random.randint(2, 10) retourne 7.
Résultat : individual = [150, 20, 7, 3, "sqrt", "gini"]. """)
    index = random.randint(0, len(individual) - 1)  # Choisir un paramètre à muter
    if index < 4:  # Paramètres numériques
        param_name = list(param_ranges.keys())[index]
        individual[index] = random.randint(*param_ranges[param_name])
    else:  # Paramètres catégoriques
        ("""individual = [150, 20, 4, 3, "sqrt", "gini"], index = 5 (criterion).
param_name = "criterion", options = ["gini", "entropy"].
random.choice(["gini", "entropy"]) retourne "entropy".
Résultat : individual = [150, 20, 4, 3, "sqrt", "entropy"]. """)
        param_name = list(param_ranges.keys())[index]
        individual[index] = random.choice(param_ranges[param_name])
    return individual,


toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)  # Sélection par tournoi (3 candidats)


# 📌 Fonction principale pour exécuter l’algorithme génétique
def main():
    # 📌 Paramètres de l’algorithme
    POP_SIZE = 100   # Taille de la population (ex. 50 individus)
    NGEN = 20  # Nombre de générations
    CXPB = 0.7  # Probabilité de croisement (50%)
    MUTPB = 0.4
    # Probabilité de mutation (20%)

    # 📌 Création de la population initiale
    pop = toolbox.population(n=POP_SIZE)
    print(f"\n🔍 Population initiale créée avec {POP_SIZE} individus.")

    # 📌 Liste pour suivre l’évolution de l’accuracy
    best_custom_precisions = []  # Modifié pour suivre custom_precision

    # 📌 Démarrer le chronomètre
    start_time = time.time()

    # 📌 Évaluation initiale de la population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit  # Assigner la fitness à chaque individu

    print("\n🔍 Début de l'optimisation génétique...")

    ("""Quoi ? La boucle itère NGEN fois (ex. 20 générations). 
    À chaque itération, on affiche le numéro de la génération (ex. "Génération 1", "Génération 2", etc.).""")
    for g in range(NGEN):
        print(f"\n-- Génération {g + 1} --")
        ("""toolbox.select(pop, len(pop)) : Utilise la méthode selTournament (enregistrée dans la toolbox) pour 
            sélectionner len(pop) individus (ex. 50) parmi la population actuelle pop. Le "tournoi" choisit les meilleurs
             selon leur fitness (custom_precision).
        list(map(toolbox.clone, offspring)) : Crée une copie profonde de chaque individu sélectionné.
        Pourquoi ?
        Sélection : On garde les individus les plus performants pour "reproduire" leurs caractéristiques, comme dans
        la sélection naturelle.
        Clonage : On clone pour éviter que les modifications (croisement/mutation) affectent la population originale directement.
        Exemple :
        pop = [[150, 20, 4, 3, "sqrt", "gini"], [200, 10, 5, 2, "log2", "entropy"], ...] (50 individus).
        Fitness : [0.85, 0.82, ...].
        Après sélection : offspring = [[150, 20, 4, 3, "sqrt", "gini"], [200, 10, 5, 2, "log2", "entropy"], ...]
         (les meilleurs sont choisis).
        Clonage : offspring est une copie indépendante.""")
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))  # Clonage pour éviter les modifications directes

        ("""Exemple :
child1 = [150, 20, 4, 3, "sqrt", "gini"], child2 = [200, 10, 5, 2, "log2", "entropy"].
Si random.random() = 0.3 < 0.5, on croise.
Points de coupe : entre positions 1 et 2, et 4 et 5.
Résultat :
child1 = [150, 10, 5, 2, "sqrt", "gini"]
child2 = [200, 20, 4, 3, "log2", "entropy"]
Fitness supprimée car les enfants sont nouveaux.""")
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 📌 Mutation
        ("""Exemple :
mutant = [150, 20, 4, 3, "sqrt", "gini"].
Si random.random() = 0.1 < 0.2, on mute.
Mutation sur index = 1 (max_depth) : nouvelle valeur → 25.
Résultat : mutant = [150, 25, 4, 3, "sqrt", "gini"].""")
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 📌 Évaluation des individus modifiés
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        ("""Quoi ? La population actuelle pop est remplacée par la nouvelle génération offspring.
        Pourquoi ? C’est la fin d’une génération : les nouveaux individus (sélectionnés, croisés, mutés)
         deviennent la base de la génération suivante.
        Exemple : Si offspring contient 50 individus améliorés, pop est mise à jour avec ces 50 nouveaux.""")
        # 📌 Remplacement de la population
        pop[:] = offspring

        # 📌 Statistiques et affichage du meilleur individu
        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = tools.selBest(pop, k=1)[0]  # Meilleur individu de la génération

        # Calcul de la custom_precision pour le meilleur individu (via fitness)
        custom_precision_score = best_ind.fitness.values[0]  # Récupéré directement de la fitness (CV)
        best_custom_precisions.append(custom_precision_score)

        # Calcul de l’accuracy pour le meilleur individu (inchangé pour l’affichage)
        model = RandomForestClassifier(
            n_estimators=best_ind[0], max_depth=best_ind[1], min_samples_split=best_ind[2],
            min_samples_leaf=best_ind[3], max_features=best_ind[4], criterion=best_ind[5],
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred_train)

        # 📌 Affichage détaillé
        print(f"  Meilleurs hyperparamètres : {best_ind}")
        print(f"  Score personnalisé (fitness) : {best_ind.fitness.values[0]:.4f}")
        print(f"  Accuracy (train) : {accuracy:.4f}")
        print(f"  Moyenne des scores : {sum(fits) / len(pop):.4f}")
        print(f"  Minimum : {min(fits):.4f}")
        print(f"  Maximum : {max(fits):.4f}")

    # 📌 Calcul et affichage du temps d’exécution
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Temps d'exécution de l'algorithme génétique : {elapsed_time / 60:.2f} minutes")

    # 📌 Résultat final : meilleur individu
    best_individual = tools.selBest(pop, k=1)[0]
    print("\n✅ Meilleurs hyperparamètres finaux :", best_individual)

    # 📌 Entraînement du modèle final
    best_model = RandomForestClassifier(
        n_estimators=best_individual[0], max_depth=best_individual[1],
        min_samples_split=best_individual[2], min_samples_leaf=best_individual[3],
        max_features=best_individual[4], criterion=best_individual[5],
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # 📌 Évaluation sur train et test
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    custom_score_train = custom_precision(y_train, y_pred_train)
    custom_score_test = custom_precision(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # 📌 Affichage des résultats finaux
    print("\n📊 Résultats finaux :")
    print(f"🔹 Score personnalisé (train) : {custom_score_train:.4f}")
    print(f"🔹 Accuracy (train) : {accuracy_train:.4f}")
    print(f"🔹 Score personnalisé (test) : {custom_score_test:.4f}")
    print(f"🔹 Accuracy (test) : {accuracy_test:.4f}")

    # 📌 Graphique de l’évolution de la custom_precision
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NGEN + 1), best_custom_precisions, marker='o', color='blue', linestyle='-', label='Custom Precision (validation croisée)')
    plt.xlabel('Génération')
    plt.ylabel('Custom Precision Moyenne (validation)')
    plt.title('Évolution de la Custom Precision au fil des générations (Algorithme Génétique)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_individual


# 📌 Lancement de l’algorithme
if __name__ == "__main__":
    best_solution = main()
    print("\n🎯 Optimisation terminée avec succès !")