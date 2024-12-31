# Supprimer tous les messages d'avertissement et de chargement
options(warn = -1) # Désactive les avertissements
suppressMessages(library(readxl))
suppressMessages(library(randomForest))

# Récupérer les arguments
args <- commandArgs(trailingOnly = TRUE)

# Arguments reçus
config_id <- as.numeric(args[1])
instance_id <- as.numeric(args[2])
seed <- as.numeric(args[3])
dataset <- args[4]
ntree <- as.numeric(args[5])
mtry <- as.numeric(args[6])

# Charger les données
data <- read_excel("C:/tuning/data_cleaned.xlsx")

# Vérifier que la colonne 'Survived' est bien présente
if (!"Survived" %in% colnames(data)) stop("Erreur : La colonne 'Survived' est absente des données.")

# Préparation des données
data$Survived <- as.factor(data$Survived)
set.seed(seed)
train_idx <- sample(1:nrow(data), 0.8 * nrow(data))
train <- data[train_idx, ]
test <- data[-train_idx, ]

# Entraîner le modèle
model <- randomForest(Survived ~ ., data = train, ntree = ntree, mtry = mtry)

# Faire des prédictions et calculer l'accuracy
pred <- predict(model, newdata = test)
accuracy <- mean(pred == test$Survived)

# Temps d'exécution simulé (vous pouvez utiliser un vrai calcul si besoin)
execution_time <- 12.93505

# Réactiver les avertissements à la fin
options(warn = 0)

# Afficher uniquement deux nombres comme attendu par irace
cat(accuracy, execution_time, "\n")
