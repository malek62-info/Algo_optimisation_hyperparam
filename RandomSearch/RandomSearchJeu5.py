import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, make_scorer
from scipy.stats import randint

# 📌 Fonction de métrique personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2

data_train_path = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/A_instance.xlsx"
result_file = "RandomSearchRes5.txt"

# 📌 Chargement des données
df_train = pd.read_excel(data_train_path)
X_train = df_train.drop(columns=['target'])
y_train = df_train['target']


# 📌 Définition des distributions de paramètres
param_dist = {
    'n_estimators': randint(10, 201),
    'max_depth': randint(5, 31),
    'min_samples_split': randint(2, 13),
    'min_samples_leaf': randint(1, 12),
    'max_features': ["sqrt", "log2"],
    'criterion': ["gini", "entropy"],
    'class_weight': ['balanced', 'balanced_subsample'],
    'ccp_alpha': [0.003]
}

# 📌 Fonction de sampling
def sample_params(dist_dict):
    return {
        key: dist.rvs() if hasattr(dist, 'rvs') else random.choice(dist)
        for key, dist in dist_dict.items()
    }

# 📌 Initialisation
best_score = -float('inf')
best_params = None
start_time = time.time()
time_limit = 7200  # 2 heures
n_iter = 50000
custom_scorer = make_scorer(custom_precision)
score_history = []

print("🔍 Début de la recherche aléatoire des hyperparamètres...\n")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 📌 Boucle de recherche aléatoire
for i in range(n_iter):
    params = sample_params(param_dist)
    model = RandomForestClassifier(random_state=42, **params)

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=custom_scorer, n_jobs=-1)
    mean_score = np.mean(cv_results['test_score'])  # ✅ Corrigé ici
    score_history.append(mean_score)

    elapsed_time = time.time() - start_time

    # Mise à jour si meilleur
    if mean_score > best_score and elapsed_time < time_limit:
        best_score = mean_score
        best_params = params

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Best Score: {best_score:.4f}\n")
            for key, val in best_params.items():
                f.write(f"{key}: {val}\n")

        print(f"✅ Itération {i+1} | Nouveau meilleur score : {best_score:.4f}")
        print(f"   ➤ Params : {best_params}\n")

    print(f"⏳ Itération {i+1}/{n_iter} | Score : {mean_score:.4f} | Temps écoulé : {elapsed_time / 60:.2f} min")

    if elapsed_time >= time_limit:
        print("\n🛑 Limite de temps atteinte. Arrêt de la recherche.")
        break

# 📌 Résumé final
total_time = time.time() - start_time
print(f"\n✅ Recherche terminée en {total_time / 60:.2f} minutes.")
if best_params:
    print(f"🔝 Meilleur score trouvé : {best_score:.4f}")
else:
    print("⚠️ Aucune configuration optimale n'a pu être trouvée dans le temps imparti.")

# 📊 Affichage de l'évolution du score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(score_history) + 1), score_history, marker='o', linestyle='-', color='blue')
plt.title("Évolution du score custom_precision au fil des itérations")
plt.xlabel("Itérations")
plt.ylabel("Score Moyen (custom_precision)")
plt.grid(True)
plt.legend(["Custom Precision"])
plt.tight_layout()
plt.show()
