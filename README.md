# Algo_optimisation_hyperparam

Hyperparameter Optimization Algorithms This repository provides implementations of various hyperparameter optimization and variable selection algorithms, including: Grid Search Random Search Genetic Algorithms Bayesian Optimization irace These algorithms are designed to improve feature selection and hyperparameter tuning for machine learning models, ensuring enhanced performance and efficiency.
_______________________________________________________________________________________________________________________________________________________________________________________________________________________

Algorithmes d'Optimisation des Hyperparamètres Ce dépôt propose des implémentations de différents algorithmes d'optimisation des hyperparamètres et de sélection de variables, notamment : Recherche exhaustive (Grid Search) Recherche aléatoire (Random Search) Algorithmes génétiques Optimisation bayésienne Ces algorithmes visent à améliorer la sélection de variables et le réglage des hyperparamètres pour les modèles d'apprentissage automatique, garantissant une performance et une efficacité accrues.

_____________________________________________________________________________________________________________________________________________________________________________________________________________________

# Explication de l'algorithme Grid Search

 1-Chargement des données :
Nous commençons par charger un ensemble de données qui sera utilisé pour entraîner et tester le modèle.

 2-Entraînement initial avec des hyperparamètres aléatoires :
Une première Random Forest (forêt aléatoire) est entraînée avec des hyperparamètres définis de manière arbitraire. Cela sert de point de comparaison pour évaluer les améliorations apportées par l'optimisation.

 3-Optimisation avec Grid Search :
L'algorithme Grid Search est utilisé pour trouver la meilleure combinaison d'hyperparamètres.
Dans cet exemple, nous avons optimisé trois hyperparamètres :
Le nombre d'arbres (n_estimators)
La profondeur maximale de l'arbre (max_depth)
Le nombre minimum d'échantillons requis pour diviser un nœud (min_samples_split)
Vous pouvez ajouter d'autres hyperparamètres selon vos besoins.

 4-Entraînement avec les meilleurs hyperparamètres :
Une nouvelle Random Forest est ensuite entraînée en utilisant les meilleurs hyperparamètres identifiés par Grid Search. Cela permet d'obtenir un modèle optimisé.

 5-Visualisation des résultats :
plusieurs graphiques sont affichés :
Une comparaison de la précision du modèle avant et après l'optimisation.
Le nombre total de combinaisons d'hyperparamètres testées par Grid Search.
Les scores moyens obtenus pour chaque combinaison d'hyperparamètres testée.
