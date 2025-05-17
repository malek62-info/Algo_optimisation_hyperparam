import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score

# Chargement des données
chemin_train = r"C:\Users\malle\OneDrive\Bureau\Algo_optimisation_hyperparam\als_train_t3.xlsx"
chemin_test = r"C:\Users\malle\OneDrive\Bureau\Algo_optimisation_hyperparam\als_test_t3.xlsx"

donnees_train = pd.read_excel(chemin_train)
donnees_test = pd.read_excel(chemin_test)

X_train = donnees_train.iloc[:, :-1]
y_train = donnees_train.iloc[:, -1]
X_test = donnees_test.iloc[:, :-1]
y_test = donnees_test.iloc[:, -1]

# Liste des hyperparamètres
hyperparams_list = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 3,
     "max_features": "sqrt", "criterion": "gini", "class_weight": "balanced", "ccp_alpha": 0.004},

    {"n_estimators": 140, "max_depth": 2, "min_samples_split": 12, "min_samples_leaf": 1,
     "max_features": "log2", "criterion": "entropy", "class_weight": "balanced", "ccp_alpha": 0.003},

    {"n_estimators": 75, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2,
     "max_features": "sqrt", "criterion": "log_loss", "class_weight": "balanced_subsample", "ccp_alpha": 0.004},

    {"n_estimators": 200, "max_depth": 1, "min_samples_split": 3, "min_samples_leaf": 5,
     "max_features": "log2", "criterion": "entropy", "class_weight": "balanced", "ccp_alpha": 0.003},

    {"n_estimators": 50, "max_depth": 8, "min_samples_split": 10, "min_samples_leaf": 4,
     "max_features": "sqrt", "criterion": "gini", "class_weight": "balanced_subsample", "ccp_alpha": 0.004}
]

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score

# Chargement des données
chemin_train = r"C:\Users\malle\OneDrive\Bureau\Algo_optimisation_hyperparam\als_train_t3.xlsx"
chemin_test = r"C:\Users\malle\OneDrive\Bureau\Algo_optimisation_hyperparam\als_test_t3.xlsx"

donnees_train = pd.read_excel(chemin_train)
donnees_test = pd.read_excel(chemin_test)

X_train = donnees_train.iloc[:, :-1]
y_train = donnees_train.iloc[:, -1]
X_test = donnees_test.iloc[:, :-1]
y_test = donnees_test.iloc[:, -1]

# Liste des hyperparamètres
hyperparams_list = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 3,
     "max_features": "sqrt", "criterion": "gini", "class_weight": "balanced", "ccp_alpha": 0.004},

    {"n_estimators": 140, "max_depth": 2, "min_samples_split": 12, "min_samples_leaf": 1,
     "max_features": "log2", "criterion": "entropy", "class_weight": "balanced", "ccp_alpha": 0.003},

    {"n_estimators": 75, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2,
     "max_features": "sqrt", "criterion": "log_loss", "class_weight": "balanced_subsample", "ccp_alpha": 0.004},

    {"n_estimators": 200, "max_depth": 1, "min_samples_split": 3, "min_samples_leaf": 5,
     "max_features": "log2", "criterion": "entropy", "class_weight": "balanced", "ccp_alpha": 0.003},

    {"n_estimators": 1, "max_depth": 1 , "min_samples_split": 2, "min_samples_leaf": 4,
     "max_features": "sqrt", "criterion": "gini", "class_weight": "balanced_subsample", "ccp_alpha": 0.005}
]

# Test et affichage
for i, params in enumerate(hyperparams_list, 1):
    start = time.time()
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n🔧 Modèle {i} — Hyperparamètres : {params}")
    print(f"⏱️  Temps d'exécution : {time.time() - start:.2f} sec")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"🎯 F1-score : {f1:.4f}")
    print(f"🔁 Recall   : {recall:.4f}")
