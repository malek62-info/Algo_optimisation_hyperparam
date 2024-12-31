import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from skopt import BayesSearchCV 
import matplotlib.pyplot as plt
import time  

# Définir votre fonction de précision personnalisée
def custom_precision(y_true, y_pred):
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    average_recall = (recall_class_1 + recall_class_0) / 2
    return average_recall

# Charger les données depuis un fichier Excel
df = pd.read_excel('C:/irace_random_forest/data_cleaned.xlsx')

print(df.head())

X = df.drop(columns=['Survived'])  
y = df['Survived']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators_value = 6  
max_depth_value = 2  
min_samples_split_value = 5 
min_samples_leaf_value = 3 
max_features_value = 'sqrt' 

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

# Afficher l'importance des variables
importances = rf_random.feature_importances_
indices = np.argsort(importances)[::-1]  

selected_features = X.columns[indices[:10]]  
print(f"Les 10 variables les plus importantes sont : {selected_features}")
print(f"Importance des variables : {importances[indices[:10]]}")

y_pred_random = rf_random.predict(X_test)

precision_random = custom_precision(y_test, y_pred_random)
print(f"Précision avant optimisation (modèle aléatoire) : {precision_random:.4f}")

param_dist = {
    'n_estimators': (50, 100, 150, 200),  # Nombre d'arbres aléatoires
    'max_depth': (None, 10, 20, 30, 40, 50),  # Profondeur des arbres
    'min_samples_split': (2, 5, 10, 20, 30),  # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': (1, 2, 4, 8, 10),  # Nombre minimum d'échantillons dans une feuille
    'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 0.8],  # Nombre de caractéristiques à tester à chaque split
    'bootstrap': [True, False],  # Utilisation de l'échantillonnage avec ou sans remise
    'criterion': ['gini', 'entropy'],  # Critère de division des nœuds (Gini ou Entropie)
    'class_weight': ['balanced', 'balanced_subsample', None],  # Poids des classes
}

start_time = time.time()

bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_dist,
    n_iter=50,  # Nombre d'itérations pour une meilleure exploration
    cv=5,
    verbose=3,
    n_jobs=-1,
    random_state=42
)

# Entraîner le modèle avec l'optimisation bayésienne
bayes_search.fit(X_train, y_train)

# Calculer le temps écoulé pour l'optimisation bayésienne
execution_time = time.time() - start_time
print(f"Temps d'exécution de l'optimisation bayésienne : {execution_time:.4f} secondes")

# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres après optimisation bayésienne : ", bayes_search.best_params_)

# Étape 3 : Sélection de variables avec le modèle optimisé
best_rf = bayes_search.best_estimator_

rf_optimized = RandomForestClassifier(
    random_state=42,
    n_estimators=bayes_search.best_params_['n_estimators'],
    max_depth=bayes_search.best_params_['max_depth'],
    min_samples_split=bayes_search.best_params_['min_samples_split'],
    min_samples_leaf=bayes_search.best_params_['min_samples_leaf'],
    max_features=bayes_search.best_params_['max_features'],
    bootstrap=bayes_search.best_params_['bootstrap'],
    criterion=bayes_search.best_params_['criterion'],
    class_weight=bayes_search.best_params_['class_weight']
)

# Entraîner le modèle optimisé
rf_optimized.fit(X_train, y_train)

importances_best = rf_optimized.feature_importances_
indices_best = np.argsort(importances_best)[::-1]

# Prédiction avec le modèle optimisé
y_pred_best = rf_optimized.predict(X_test)

precision_best = custom_precision(y_test, y_pred_best)
print(f"Précision après optimisation (modèle optimisé) : {precision_best:.4f}")

results = {
    'before_optimization': {
        'precision': precision_random,
        'selected_features': selected_features,
        'importances': importances[indices[:10]]
    },
    'after_optimization': {
        'precision': precision_best,
        'selected_features': selected_features_best,
        'importances': importances_best[indices_best[:10]]
    }
}

precisions = [precision_random, precision_best]
labels = ['Avant optimisation', 'Après optimisation']

plt.bar(labels, precisions, color=['blue', 'green'])
plt.ylabel('Précision')
plt.title('Précision avant et après optimisation bayésienne')
plt.show()


results_df = pd.DataFrame(bayes_search.cv_results_)

plt.figure(figsize=(12, 8))
plt.scatter(results_df['param_n_estimators'], results_df['param_max_depth'], c=results_df['mean_test_score'], cmap='viridis')
plt.colorbar(label='Mean Test Score (Accuracy)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.title('Accuracy de chaque combinaison d\'hyperparamètres')
plt.show()

results
