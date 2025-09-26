# Load required libraries
library(caret)
library(dplyr)
library(glmnet)
library(xgboost)

# Step 1: Load and preprocess data
d <- read.csv("C:/Users/Hp User/Desktop/grades.csv", stringsAsFactors = FALSE)
d <- d[, colSums(is.na(d)) < nrow(d)]  # Remove empty columns

# Step 2: Feature extraction
extract_simple <- function(data, pattern, prefix) {
  cols <- grep(pattern, names(data), value = TRUE, ignore.case = TRUE)
  for (i in seq_along(cols)) {
    data[[paste0(prefix, i)]] <- data[[cols[i]]]
  }
  data[, !names(data) %in% cols]
}

d <- extract_simple(d, "Exams|X$", "Exam")
d <- extract_simple(d, "Homework", "HW")
d <- extract_simple(d, "Compulsory", "Comp")
d <- extract_simple(d, "Optional", "Opt")

# Step 3: Clean and engineer features
d_clean <- d[2:161, ] %>%
  select(where(is.numeric)) %>%
  mutate(
    Exam_Avg = rowMeans(select(., starts_with("Exam")), na.rm = TRUE),
    HW_Avg = rowMeans(select(., starts_with("HW")), na.rm = TRUE),
    HW_Completion = rowSums(!is.na(select(., starts_with("HW")))) / length(grep("HW", names(.)))
  ) %>%
  replace(is.na(.), 0)

# Step 4: Create target variable
d_clean$Final_Score <- rowMeans(d_clean, na.rm = TRUE)

# Step 5: Train-test split
set.seed(123)
train_index <- createDataPartition(d_clean$Final_Score, p = 0.8, list = FALSE)
train_data <- d_clean[train_index, ]
test_data <- d_clean[-train_index, ]

# Step 6: Baseline Linear Regression model
simple_model <- train(
  Final_Score ~ Exam_Avg + HW_Avg + HW_Completion,
  data = train_data,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 5)
)

predictions <- predict(simple_model, newdata = test_data)

results_linear <- data.frame(
  Model = "Linear Regression",
  MAE = mean(abs(test_data$Final_Score - predictions)),
  RMSE = sqrt(mean((test_data$Final_Score - predictions)^2)),
  R2 = cor(test_data$Final_Score, predictions)^2
)

print(results_linear)

# Plot: Linear Regression diagnostics
plot(test_data$Final_Score, predictions,
     main = "Linear Model: Actual vs Predicted",
     xlab = "Actual Final Score", ylab = "Predicted Final Score")
abline(0, 1, col = "red")

# Step 7: Feature selection (RFE)
control <- rfeControl(functions = lmFuncs, method = "cv", number = 5)
rfe_model <- rfe(
  train_data[, -which(names(train_data) == "Final_Score")],
  train_data$Final_Score,
  sizes = c(1:10),
  rfeControl = control
)
print(rfe_model)
plot(rfe_model, type = c("g", "o"))

# Step 8: XGBoost with all features
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  eta = c(0.05, 0.1),
  max_depth = c(3, 6),
  gamma = c(0, 1),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

xgb_model <- train(
  Final_Score ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = xgb_grid,
  verbose = FALSE,
  verbosity = 0
)

xgb_preds <- suppressWarnings(
  predict(xgb_model$finalModel, 
          newdata = as.matrix(test_data[ , -which(names(test_data) == "Final_Score")]),
          iteration_range = c(0, xgb_model$bestTune$nrounds))
)

results_xgb <- data.frame(
  Model = "XGBoost (All Features)",
  MAE = mean(abs(test_data$Final_Score - xgb_preds)),
  RMSE = sqrt(mean((test_data$Final_Score - xgb_preds)^2)),
  R2 = cor(test_data$Final_Score, xgb_preds)^2
)

print(results_xgb)

# Plot: XGBoost (all features) diagnostics
plot(test_data$Final_Score, xgb_preds,
     main = "XGBoost (All Features): Actual vs Predicted",
     xlab = "Actual Final Score", ylab = "Predicted Final Score")
abline(0, 1, col = "blue")

# Step 9: XGBoost using RFE-selected features

# Extract selected features from RFE
selected_features <- predictors(rfe_model)
print(selected_features)

# Subset data
train_selected <- train_data[, c(selected_features, "Final_Score")]
test_selected <- test_data[, c(selected_features, "Final_Score")]

# Train XGBoost again with selected features
xgb_model_rfe <- train(
  Final_Score ~ .,
  data = train_selected,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = xgb_grid,
  verbose = FALSE
)

# Predict
xgb_preds_rfe <- suppressWarnings(
  predict(xgb_model_rfe$finalModel,
          newdata = as.matrix(test_selected[, selected_features]),
          iteration_range = c(0, xgb_model_rfe$bestTune$nrounds))
)

# Evaluate
results_xgb_rfe <- data.frame(
  Model = "XGBoost (RFE)",
  MAE = mean(abs(test_selected$Final_Score - xgb_preds_rfe)),
  RMSE = sqrt(mean((test_selected$Final_Score - xgb_preds_rfe)^2)),
  R2 = cor(test_selected$Final_Score, xgb_preds_rfe)^2
)

print(results_xgb_rfe)

# Plot: XGBoost (RFE) diagnostics
plot(test_selected$Final_Score, xgb_preds_rfe,
     main = "XGBoost (RFE): Actual vs Predicted",
     xlab = "Actual Final Score", ylab = "Predicted Final Score")
abline(0, 1, col = "darkgreen")

