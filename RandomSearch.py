# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# Définir votre fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    average_recall = (recall_class_1 + recall_class_0) / 2
    return average_recall

# Charger les données depuis un fichier Excel
df = pd.read_excel('C:/irace_random_forest/data_cleaned.xlsx')

# Vérifier les premières lignes pour comprendre la structure des données
print(df.head())

# Sélectionner les variables (X) et la cible (y)
# Remplacer 'Survived' par le nom réel de votre colonne cible
X = df.drop(columns=['Survived'])  # Remplacer 'Survived' par votre colonne cible
y = df['Survived']  # Remplacer 'Survived' par votre colonne cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir directement les valeurs aléatoires pour les hyperparamètres
n_estimators_value = 6  # Exemple d'un nombre d'arbres
max_depth_value = 2  # Profondeur maximale des arbres
min_samples_split_value = 5  # Nombre minimum d'échantillons pour diviser un nœud
min_samples_leaf_value = 3  # Nombre minimum d'échantillons dans une feuille
max_features_value = 'sqrt'  # Nombre de caractéristiques à tester à chaque split


# Initialisation du RandomForest avec ces paramètres définis
rf_random = RandomForestClassifier(
    n_estimators=n_estimators_value,
    max_depth=max_depth_value,
    min_samples_split=min_samples_split_value,
    min_samples_leaf=min_samples_leaf_value,
    max_features=max_features_value,
    
    random_state=42  # Pour garantir la reproductibilité
)

# Entraîner le modèle
rf_random.fit(X_train, y_train)


# Prédiction avec le modèle random
y_pred_random = rf_random.predict(X_test)

# Calculer la précision avec la fonction custom_precision
precision_random = custom_precision(y_test, y_pred_random)
print(f"Précision avant optimisation (modèle aléatoire) : {precision_random:.4f}")

# Étape 2 : Optimisation des hyperparamètres avec RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50, 200, 10),  # Nombre d'arbres aléatoires
    'max_depth': [None, 10, 20, 30, 40, 50],  # Profondeur des arbres
    'min_samples_split': [2, 5, 10, 20, 30],  # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': [1, 2, 4, 8, 10],  # Nombre minimum d'échantillons dans une feuille
    'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 0.8],  # Nombre de caractéristiques à tester à chaque split
    'bootstrap': [True, False],  # Utilisation de l'échantillonnage avec ou sans remise
    'criterion': ['gini', 'entropy'],  # Critère de division des nœuds (Gini ou Entropie)
    'class_weight': ['balanced', 'balanced_subsample', None],  # Poids des classes
}


# Définir RandomizedSearchCV avec plus d'itérations et plus de plis
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=30,  # Augmenter le nombre d'itérations pour plus d'exploration
                                   cv=10,  # Utiliser 10 plis pour la validation croisée pour une évaluation plus robuste
                                   verbose=3,  
                                   n_jobs=-1,  
                                   random_state=42)

# Entraîner le modèle avec la recherche aléatoire
random_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres après RandomizedSearch : ", random_search.best_params_)

# Étape 3 : Sélection de variables avec le modèle optimisé
best_rf = random_search.best_estimator_

# Refaire la sélection de variables avec les meilleurs hyperparamètres
best_rf.fit(X_train, y_train)
importances_best = best_rf.feature_importances_
indices_best = np.argsort(importances_best)[::-1]


# Prédiction avec le modèle optimisé
y_pred_best = best_rf.predict(X_test)

# Calculer la précision avec la fonction custom_precision
precision_best = custom_precision(y_test, y_pred_best)
print(f"Précision après optimisation (modèle optimisé) : {precision_best:.4f}")

# Afficher la comparaison de la précision avant et après optimisation
precisions = [precision_random, precision_best]
labels = ['Avant optimisation', 'Après optimisation']

plt.bar(labels, precisions, color=['blue', 'green'])
plt.ylabel('Précision')
plt.title('Précision avant et après optimisation')
plt.show()
