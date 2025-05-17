import sys
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score

args = sys.argv[1:]
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

data = pd.read_excel(params["dataset"])
if "target" not in data.columns:
    raise ValueError("La colonne cible 'target' est absente du dataset")

# Séparation des features et de la cible
X = data.drop(columns=["target"])
y = data["target"]

def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    return (recall_class_1 + recall_class_0) / 2  # Moyenne des recalls

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

start_time = time.time()
cv_scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(custom_precision))
mean_cv_score = cv_scores.mean()
exec_time = time.time() - start_time

print(f"{- mean_cv_score:.6f} {exec_time:.2f}")
