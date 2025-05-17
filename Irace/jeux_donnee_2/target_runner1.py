import sys
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score

# 📌 Récupération des arguments passés par irace
args = sys.argv[1:]  # sys.argv[0] est le nom du script, donc on prend les suivants

# 📌 Mapping des paramètres
params = {
    "config_id": int(args[0]),
    "instance_id": int(args[1]),
    "seed": int(args[2]),
    "dataset": args[3],
    "n_estimators": int(args[4]),
    "max_depth": int(args[5]),
    "min_samples_split": int(args[6]),
    "min_samples_leaf": int(args[7]),
    "max_features": args[8],
    "criterion": args[9],
    "class_weight":  args[10],
    "ccp_alpha": float(args[11])
}


# 📌 Chargement des données
data = pd.read_excel(params["dataset"])
if "target" not in data.columns:
    raise ValueError("La colonne cible 'Survived' est absente du dataset")


# Vérifier si la variable cible est sous forme de chaîne et la convertir en 0/1
if data['target'].dtype == object:
    data['target'] = data['target'].map(lambda x: 1 if x == "VRAI" else 0)

# Séparation des features et de la cible
X = data.drop(columns=["target"])
y = data["target"]

# 📌 Fonction de scoring personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2  # Moyenne des recalls

# 📌 Création du modèle Random Forest avec les hyperparamètres optimisés
model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    min_samples_split=params["min_samples_split"],
    min_samples_leaf=params["min_samples_leaf"],
    max_features=params["max_features"],
    criterion=params["criterion"],
    class_weight=params["class_weight"],
    ccp_alpha=params["ccp_alpha"],
    random_state=params["seed"],
    n_jobs=-1
)

# 📌 Validation croisée (5-fold)
start_time = time.time()
cv_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(custom_precision))
mean_cv_score = cv_scores.mean()
exec_time = time.time() - start_time

# 📌 Résultat pour irace (on minimise donc 1 - score)
print(f"{- mean_cv_score:.6f} {exec_time:.2f}")
