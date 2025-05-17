options(warn = -1)
suppressMessages({
  library(readxl)
  library(randomForest)
  library(caret)
})

# Get arguments from irace
args <- commandArgs(trailingOnly = TRUE)

# Parse parameters
params <- list(
  config_id = as.numeric(args[1]),
  instance_id = as.numeric(args[2]),
  seed = as.numeric(args[3]),
  dataset = args[4],
  n_estimators = as.numeric(args[5]),
  max_depth = ifelse(args[6] == "NULL", NULL, as.numeric(args[6])),
  min_samples_split = as.numeric(args[7]),
  min_samples_leaf = as.numeric(args[8]),
  max_features = args[9],
  criterion = args[10]
)

# Parameter validation
if (params$n_estimators < 10 || params$n_estimators > 500) {
  stop("n_estimators must be between 10 and 500")
}
if (!is.null(params$max_depth) && (params$max_depth < 3 || params$max_depth > 200)) {
  stop("max_depth must be between 3 and 200 or NULL")
}
if (params$min_samples_split < 2 || params$min_samples_split > 50) {
  stop("min_samples_split must be between 2 and 50")
}
if (params$min_samples_leaf < 1 || params$min_samples_leaf > 50) {
  stop("min_samples_leaf must be between 1 and 50")
}
if (!params$max_features %in% c("sqrt", "log2")) {
  stop("max_features must be 'sqrt' or 'log2'")
}
if (!params$criterion %in% c("gini", "entropy")) {
  stop("criterion must be 'gini' or 'entropy'")
}

# Start timing
start_time <- Sys.time()

# Load data
data <- read_excel(params$dataset)
if (!"Survived" %in% colnames(data)) {
  stop("Target column 'Survived' not found in dataset")
}

# Prepare data
data$Survived <- as.factor(data$Survived)
set.seed(params$seed)

# Custom precision function
custom_precision <- function(y_true, y_pred) {
  cm <- confusionMatrix(y_pred, y_true)
  (cm$byClass["Sensitivity"] + cm$byClass["Specificity"]) / 2
}

# Manual 5-fold cross-validation
folds <- createFolds(data$Survived, k = 5)
cv_scores <- numeric(5)

for (i in seq_along(folds)) {
  # Split data
  train <- data[-folds[[i]], ]
  valid <- data[folds[[i]], ]

  # Train model with randomForest
  model <- randomForest(
    Survived ~ .,
    data = train,
    ntree = params$n_estimators,
    maxnodes = params$max_depth,
    nodesize = params$min_samples_split,
    minbucket = params$min_samples_leaf,
    maxFeatures = params$max_features,
    criterion = params$criterion,
    importance = TRUE
  )

  # Predict on validation fold
  pred <- predict(model, newdata = valid)
  cv_scores[i] <- custom_precision(valid$Survived, pred)
}

# Calculate mean CV score
mean_cv_score <- mean(cv_scores, na.rm = TRUE)

# Calculate execution time
exec_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

# Output for irace (1 - score because irace minimizes)
cat(sprintf("%.6f %.2f\n", 1 - mean_cv_score, exec_time))

options(warn = 0)