import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


# 📌 Définir le fichier pour stocker la meilleure configuration
output_file = "bayes_optimization_results.txt"

# 📌 Chemins des fichiers de données
data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/als_train_t3.xlsx"

# 📌 Charger les données d'entraînement
df_train = pd.read_excel(data_train_path)

# 📌 Sélection des variables et de la cible
X_train = df_train.drop(columns=['Survived'])
y_train = df_train['Survived']




# 📌 Définir la fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    average_recall = (recall_class_1 + recall_class_0) / 2
    return average_recall


# 📌 Définir l'espace des hyperparamètres pour l'optimisation bayésienne
param_dist = {
    'n_estimators': (10, 200),
    'max_depth': (5, 30),
    'min_samples_split': (2, 12),
    'min_samples_leaf': (1, 12),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced','balanced_subsample'],
    'ccp_alpha': Categorical([0.003])
}



# 📌 Initialiser la meilleure valeur
best_score = -np.inf  # Score initial bas pour garantir une amélioration
best_params = None

# 📌 Liste pour stocker l'évolution de la précision personnalisée
custom_precision_history = []

# 📌 Définir la limite de temps (en secondes)
time_limit = 7200  # 2 heures
start_time = time.time()


# 📌 Fonction de callback pour surveiller le temps et sauvegarder la meilleure configuration
def on_step(optim_result):
    global best_score, best_params

    # Vérifier si le temps limite est dépassé
    elapsed_time = time.time() - start_time
    if elapsed_time > time_limit:
        print("\nTemps limite dépassé ! Affichage du graphe...")

        # 📊 Affichage du graphique final
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(custom_precision_history) + 1), custom_precision_history, marker='o', linestyle='-', color='blue',
                 label="Custom Precision")
        plt.xlabel("Itérations")
        plt.ylabel("Custom Precision")
        plt.title("Évolution de la Custom Precision (Optimisation interrompue)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return False  # Continuer l'optimisation, mais ne pas sauvegarder de nouvelles valeurs

    # Récupérer les derniers hyperparamètres testés et leur score
    current_params = dict(zip(param_dist.keys(), optim_result.x))
    current_score = -optim_result.fun  # Négatif car BayesSearchCV minimise

    custom_precision_history.append(current_score)  # Stocke l'évolution

    # Si cette configuration est meilleure, on met à jour
    if current_score > best_score:
        best_score = current_score
        best_params = current_params

        # 📌 Écrire uniquement la meilleure configuration dans le fichier
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Best Score: {best_score:.4f}\n")
            for key ,value in best_params.items():
                f.write(f" {value}\n")

        print(f"\nNouvelle meilleure configuration trouvée ! Custom Precision: {best_score:.4f}")
        print(f"Hyperparamètres: {best_params}\n")

    return False  # Continuer l'optimisation


# 📌 Configuration de la recherche bayésienne
custom_scorer = make_scorer(custom_precision)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_dist,
    n_iter=50000,
    cv=cv,
    verbose=3,
    n_jobs=-1,
    random_state=42,
    scoring=custom_scorer,
    refit=True,
    return_train_score=True
)

# 📌 Exécuter BayesSearchCV avec le callback
bayes_search.fit(X_train, y_train, callback=on_step)

# 📌 Temps total d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution : {elapsed_time / 60:.2f} minutes")

# 📌 Sauvegarde finale si dans le temps limite
if elapsed_time <= time_limit and best_params:
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\nTemps d'exécution : {elapsed_time / 60:.2f} minutes\n")
    print("\nMeilleure configuration sauvegardée !")
else:
    print("\nAucune configuration sauvegardée car le temps limite a été dépassé.")

# 📊 Affichage du graphique si l'optimisation a pris toute la durée
if elapsed_time >= time_limit:
    print("\nAffichage du graphique car le temps limite a été atteint !")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(custom_precision_history) + 1), custom_precision_history, marker='o', linestyle='-', color='blue',
             label="Custom Precision")
    plt.xlabel("Itérations")
    plt.ylabel("Custom Precision")
    plt.title("Évolution de la Custom Precision (Temps limite atteint)")
    plt.legend()
    plt.grid(True)
    plt.show()
