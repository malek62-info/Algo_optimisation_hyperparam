options(warn = -1)
suppressMessages({
  library(readxl)
  library(randomForest)
  library(caret)
})

# Récupération des arguments
args <- commandArgs(trailingOnly = TRUE)

# Initialisation des paramètres
params <- list(
  config_id = as.numeric(args[1]),
  instance_id = as.numeric(args[2]),
  seed = as.numeric(args[3]),
  dataset = args[4],
  n_estimators = as.numeric(args[5]),
  max_depth = as.numeric(args[6]),
  min_samples_split = as.numeric(args[7]),
  min_samples_leaf = as.numeric(args[8]),
  max_features = trimws(args[9]),
  criterion = trimws(args[10])
)

# Validation des paramètres
if (params$n_estimators < 10 || params$n_estimators > 500) stop("n_estimators doit être entre 10 et 500")
if (!is.null(params$max_depth) && (params$max_depth < 3 || params$max_depth > 200)) stop("max_depth doit être entre 3 et 200 ou NULL")
if (params$min_samples_split < 2 || params$min_samples_split > 50) stop("min_samples_split doit être entre 2 et 50")
if (params$min_samples_leaf < 1 || params$min_samples_leaf > 50) stop("min_samples_leaf doit être entre 1 et 50")
if (!params$max_features %in% c("sqrt", "log2")) stop("max_features doit être 'sqrt' ou 'log2'")
if (!params$criterion %in% c("gini", "entropy")) stop("criterion doit être 'gini' ou 'entropy'")

# Mesurer le temps de début
start_time <- Sys.time()

# Chargement et préparation des données
data <- read_excel(params$dataset)
if (!"target" %in% colnames(data)) stop("La colonne 'target' est absente des données")

data$target <- as.factor(data$target)
levels(data$target) <- c("0", "1")
set.seed(params$seed)

# Séparation train-test
train_idx <- createDataPartition(data$target, p = 0.8, list = FALSE)
train <- data[train_idx, ]
test <- data[-train_idx, ]

# Fonction de précision personnalisée
custom_precision <- function(y_true, y_pred) {
  cm <- confusionMatrix(y_pred, y_true)
  recall_class_1 <- cm$byClass["Sensitivity"]
  recall_class_0 <- cm$byClass["Specificity"]
  return((recall_class_1 + recall_class_0) / 2)
}

# Configuration de la validation croisée
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = FALSE,
  summaryFunction = function(data, lev = NULL, model = NULL) {
    out <- custom_precision(data$obs, data$pred)
    names(out) <- "CustomPrecision"
    out
  }
)

# Entraînement du modèle
tryCatch({
  model <- train(
    target ~ .,
    data = train,
    method = "rf",
    trControl = ctrl,
    ntree = params$n_estimators,
    maxnodes = params$max_depth,
    nodesize = params$min_samples_split,
    minbucket = params$min_samples_leaf,
    importance = TRUE,
    metric = "CustomPrecision",
    maximize = TRUE
  )

  # Extraction du score CV
  cv_score <- max(model$results$CustomPrecision, na.rm = TRUE)

  # Calcul du temps d'exécution
  exec_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  # Sortie pour irace (1 - score car irace minimise)
  cat(sprintf("%.6f %.2f\n", 1 - cv_score, exec_time))
}, error = function(e) {
  cat("1.000000 0.00\n")
  stop(e)
})

options(warn = 0)