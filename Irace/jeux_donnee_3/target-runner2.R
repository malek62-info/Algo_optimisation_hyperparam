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
  max_depth = as.numeric(args[6]),
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
if (!"target" %in% colnames(data)) {
  stop("Target column 'target' not found in dataset")
}

# Prepare data
data$target <- as.factor(data$target)
set.seed(params$seed)

# Train-test split (80-20)
train_idx <- createDataPartition(data$target, p = 0.8, list = FALSE)
train <- data[train_idx, ]
test <- data[-train_idx, ]

# Custom precision function
custom_precision <- function(y_true, y_pred) {
  cm <- confusionMatrix(y_pred, y_true)
  recall_class_1 <- cm$byClass["Sensitivity"]
  recall_class_0 <- cm$byClass["Specificity"]
  return((recall_class_1 + recall_class_0) / 2)
}

# 5-fold CV control
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

# Train model with error handling
tryCatch({
  model <- train(
    target ~ .,
    data = train,
    method = "rf",
    trControl = ctrl,
    ntree = params$n_estimators,
    maxdepth = params$max_depth,
    minsplit = params$min_samples_split,
    minbucket = params$min_samples_leaf,
    maxfeatures = params$max_features,
    criterion = params$criterion,
    metric = "CustomPrecision",  # Our custom metric
    maximize = TRUE              # We want to maximize precision
  )

  # Get CV results
  cv_score <- max(model$results$CustomPrecision, na.rm = TRUE)

  # Calculate execution time
  exec_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  # Output for irace (1 - score because irace minimizes)
  cat(sprintf("%.6f %.2f\n", 1 - cv_score, exec_time))

}, error = function(e) {
  # Return poor score if error occurs
  cat("1.000000 0.00\n")
  stop(e)
})

options(warn = 0)
