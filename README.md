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

_____________________________________________________________________________________________________________________________________________________________________________________________________________________

# Explication de l'algorithme d'optimisation RandomizedSearchCV :

1-Chargement des données :

Les données sont importées depuis un fichier Excel.
Les variables explicatives (X) et la cible (y) sont définies.
Les données sont ensuite divisées en ensembles d'entraînement et de test.

2-Entraînement initial avec des hyperparamètres aléatoires :
Un premier modèle de Random Forest est entraîné avec des hyperparamètres prédéfinis de manière arbitraire.
Ce modèle sert de référence pour évaluer les améliorations obtenues après optimisation.
La précision est calculée en utilisant une fonction personnalisée qui évalue la moyenne des rappels des classes.

3-Optimisation avec RandomizedSearchCV :
RandomizedSearch explore aléatoirement différentes combinaisons d'hyperparamètres, contrairement à Grid Search qui teste toutes les combinaisons possibles.
Parmi les hyperparamètres optimisés dans cet exemple :
Le nombre d'arbres (n_estimators)
La profondeur maximale de l'arbre (max_depth)
Le nombre minimum d'échantillons requis pour diviser un nœud (min_samples_split)
Et d'autres, comme max_features, criterion, et bootstrap.
L'algorithme effectue une validation croisée pour évaluer chaque combinaison d'hyperparamètres et détermine les meilleurs.

4-Entraînement avec les meilleurs hyperparamètres :
Une nouvelle Random Forest est entraînée avec les hyperparamètres optimaux trouvés par RandomizedSearch.
Les performances de ce modèle optimisé sont mesurées sur l'ensemble de test.

5-Visualisation des résultats :
Un graphique compare la précision du modèle avant et après optimisation.
Les meilleurs hyperparamètres trouvés sont affichés, permettant de comprendre comment le modèle a été amélioré.
___________________________________________________________________________________________________________________________________________________________________________________________________________________
# optimisation bayésienne

1-Chargement des données :
Le fichier de données data_cleaned.xlsx est chargé dans un DataFrame avec pandas. Les colonnes sont séparées en features (X) et target (y).

2-Entraînement initial du modèle :
Un modèle RandomForestClassifier est initialisé avec des hyperparamètres fixés (comme le nombre d'arbres, la profondeur des arbres, etc.) et entraîné sur un ensemble d'entraînement. Ensuite, on calcule et affiche l'importance des variables et la précision du modèle.

3-Optimisation des hyperparamètres avec BayesSearchCV :
Les hyperparamètres du modèle sont optimisés à l'aide d'une recherche bayésienne (BayesSearchCV). Cette méthode explore différents ensembles d'hyperparamètres pour trouver la meilleure combinaison qui maximise la performance du modèle.

4-Entraînement du modèle optimisé :
Après l'optimisation, le modèle est réentraîné avec les meilleurs hyperparamètres trouvés.

5-Comparaison de la performance avant et après optimisation :
La précision est mesurée avant et après optimisation, et un graphique comparant ces deux performances est généré. Les importances des variables les plus significatives sont également affichées.

6-Visualisation des résultats :
Un graphique est créé pour visualiser l'impact de l'optimisation sur la performance du modèle, et un autre graphique montre l'impact des différentes combinaisons d'hyperparamètres sur la précision du modèle.
 
____________________________________________________________________________________________________________________________________________________________________________________________________________________

# Optimisation d'Hyperparamètres avec Hyperband pour Random Forest
 
 1-Chargement des données :
Les données sont chargées depuis un fichier Excel (data_cleaned.xlsx).
La colonne Survived est utilisée comme cible (y), et les autres colonnes comme variables explicatives (X).
Les données sont divisées en deux ensembles : 80% pour l'entraînement et 20% pour le test.

2-Modèle initial :
Un modèle de forêt aléatoire est entraîné avec ses paramètres par défaut.
La précision personnalisée (moyenne du rappel des classes positives et négatives) est calculée pour évaluer la performance initiale.

3-Optimisation avec Hyperband :
Hyperband génère des configurations d'hyperparamètres aléatoires (exemple : nombre d'arbres, profondeur maximale des arbres, etc.).
Chaque configuration est testée par étapes : les configurations les moins performantes sont progressivement éliminées, tandis que les meilleures continuent avec plus de ressources (nombre d'itérations).

4-Modèle optimisé :
Les meilleurs hyperparamètres trouvés par Hyperband sont utilisés pour entraîner un modèle optimisé.
La performance du modèle optimisé est comparée à celle du modèle initial.

5-Visualisation :
Un graphique compare les précisions avant et après optimisation.

