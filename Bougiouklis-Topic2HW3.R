#a
# Step 1: Read the data into a data frame
data <- read.csv(file.choose())

# Step 2: Select only the numeric variables
numeric_vars <- data[sapply(data, is.numeric)]

# Step 3: Create a new data frame called house_prices_numeric
house_prices_numeric <- numeric_vars

# Step 4: Standardize the data (z-score)
house_prices_numeric_standardized <- as.data.frame(scale(house_prices_numeric))

# Step 5: Create a correlation matrix
correlation_matrix <- cor(house_prices_numeric_standardized)

# Step 6: Complete the remaining values in the table
# Assuming the table requires the mean, standard deviation, and correlation matrix
mean_values <- colMeans(house_prices_numeric)
std_dev_values <- apply(house_prices_numeric, 2, sd)

# Print the results
print("Mean values:")
print(mean_values)

print("Standard deviation values:")
print(std_dev_values)

print("Correlation matrix:")
print(correlation_matrix)

#b
# Step 1: Extract correlations with 'price'
price_correlations <- correlation_matrix["price", ]

# Step 2: Rank correlations in descending order and select top 3
sorted_correlations <- sort(price_correlations, decreasing = TRUE)
top_3_features <- names(sorted_correlations)[2:4]  # Exclude 'price' itself (rank 1)

# Step 3: Store the top 3 features in selected_features
selected_features <- top_3_features

# Step 4: Plot the correlations of the top 3 features in a bar chart
library(ggplot2)
# Create a data frame for plotting
plot_data <- data.frame(
  Feature = selected_features,
  Correlation = sorted_correlations[2:4]  # Exclude 'price' itself
)

# Plot the bar chart
ggplot(plot_data, aes(x = reorder(Feature, -Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Top 3 Features Correlated with Price",
    x = "Feature",
    y = "Correlation"
  ) +
  theme_minimal()

#c
# Step 1: Create a linear regression model
# Combine the top 3 features and the target variable into a formula
target_variable <- "price"
formula <- as.formula(paste(target_variable, "~", paste(top_3_features, collapse = " + ")))

# Fit the linear regression model
linear_model <- lm(formula, data = house_prices_numeric_standardized)

# Step 2: Extract the R-squared value
r_squared <- summary(linear_model)$r.squared

# Step 3: Calculate the Mean Absolute Error (MAE)
predictions <- predict(linear_model, house_prices_numeric_standardized)
mae <- mean(abs(house_prices_numeric_standardized[[target_variable]] - predictions))

# Step 4: Report the results
print("Linear Regression Model Summary:")
print(summary(linear_model))

print(paste("R-squared value:", r_squared))
print(paste("Mean Absolute Error (MAE):", mae))

#d
# Get mean and standard deviation of the original 'price' variable 
price_mean <- mean(numeric_vars$price, na.rm = TRUE) 
price_sd <- sd(numeric_vars$price, na.rm = TRUE) 

# De-normalize the predictions and actual values
denormalized_predictions <- (predictions * price_sd ) + price_mean
denormalized_actual <- (house_prices_numeric_standardized$price * price_sd ) + price_mean
# Provide de-normalized predictions for specific houses 
specific_house_ids <- c(5, 100, 305) 
specific_denormalized_predictions <- round(denormalized_predictions[specific_house_ids] , 3)
print(specific_denormalized_predictions)

# Plot predicted vs actual values
library(ggplot2)

# Create a data frame for plotting
plot_data1 <- data.frame(
  Actual = denormalized_actual,
  Predicted = denormalized_predictions
)

# Plot the predicted vs actual values
ggplot(plot_data1, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +  # Scatter plot of actual vs predicted
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +  # y = x line
  labs(
    title = "Predicted vs Actual Prices",
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  theme_minimal()


