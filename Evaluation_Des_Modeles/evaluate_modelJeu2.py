import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score
import warnings
import os

# Désactiver les UserWarnings (y compris celui concernant les cœurs physiques)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# 📌 Chargement des données
chemin_train = "C:/Users/malle/OneDrive/Bureau/Algo_optimisation_hyperparam/B_instance.xlsx"
df_train = pd.read_excel(chemin_train)


# Vérifier si la variable cible est en format chaîne et la convertir
if df_train['target'].dtype == object:
    df_train['target'] = df_train['target'].map(lambda x: 1 if x == "VRAI" else 0)

# 📌 Séparation des variables
X_train = df_train.drop(columns=['target'])
y_train = df_train['target']

best_params = {
    "n_estimators": 199,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 9,
    "max_features": "sqrt",
    "criterion": "entropy",
    "class_weight": "balanced_subsample",
    "ccp_alpha": 0.004
}

# 📌 Modèle avec les meilleurs hyperparamètres
model = RandomForestClassifier(**best_params, random_state=42)

# 🔁 Validation croisée à 5 plis sur les données d'entraînement
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n📊 Moyenne des accuracies sur 5 validations croisées (train) : {cv_scores.mean():.4f}")

# 📌 Entraînement final du modèle sur tout le training set
model.fit(X_train, y_train)

# 📌 Prédictions sur le train uniquement
y_pred_train = model.predict(X_train)

# 📊 Évaluation
def afficher_scores(y_true, y_pred, dataset=""):
    print(f"\n📈 Scores sur {dataset} :")
    print("🔹 Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("🔹 F1-score :", round(f1_score(y_true, y_pred), 4))
    print("🔹 Recall   :", round(recall_score(y_true, y_pred), 4))

afficher_scores(y_train, y_pred_train, "Train")
