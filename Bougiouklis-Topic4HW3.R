#a
# Step 1: Read the data
data_0 <- read.csv(file.choose())

# Step 2: Inspect the structure of the data
str(data_0)

# Step 3: Exclude the first two columns
data_clean <- data[, -c(1, 2)]

# Step 4: Check and keep only numeric columns
data_clean <- data_clean[, sapply(data_clean, is.numeric)]

# Step 5: Check the number of features in data_clean
num_features <- ncol(data_clean)
print(paste("Number of features in data_clean:", num_features))

# Step 6: Define a function for min-max normalization
min_max_normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Step 7: Apply min-max normalization to all features
scaled_data <- as.data.frame(lapply(data_clean, min_max_normalize))

# Step 8: Load necessary library for plotting
library(ggplot2)
library(reshape2)

# Step 9 : Calculate the correlation matrix
cor_matrix <- cor(scaled_data)

# Step 10: Melt the correlation matrix for ggplot
melted_cor_matrix <- melt(cor_matrix)

# Step 11: Plot the heatmap
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap", x = "", y = "", fill = "Correlation")

#b
# Step 1: Load necessary libraries
library(ggplot2)

# Step 2: Initialize a vector to store WCSS values
wcss <- numeric(10)

# Step 3: Calculate WCSS for k = 1 to k = 10
for (k in 1:10) {
#Step 4: Perform k-means clustering
  kmeans_result <- kmeans(scaled_data, centers = k, nstart = 10)
  
#Step 5 : Store the WCSS value
  wcss[k] <- kmeans_result$tot.withinss
}

#Step 6: Create a dataframe for plotting
wcss_data <- data.frame(k = 1:10, WCSS = wcss)

#Step 7: Plot WCSS vs k
ggplot(wcss_data, aes(x = k, y = WCSS)) +
  geom_line(color = "blue", size = 1.2) +
  geom_point(color = "red", size = 3) +
  labs(title = "Elbow Method for Optimal k",
       x = "Number of Clusters (k)",
       y = "Within-Cluster Sum of Squares (WCSS)") +
  theme_minimal()

#c
# Apply K-Means

set.seed(123) # Set a random seed for reproducibility
kmeans_model <- kmeans(scaled_data, centers =5, nstart = 10) 
kmeans_model

#d
# Step 1 : Perform PCA
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Step 2 : Extract the first 2 principal components
pca_data <- as.data.frame(pca_result$x[, 1:2])

# Step 3 : Apply K-Means on PCA-reduced data
set.seed(456)  # Set a random seed for reproducibility
pca_kmeans <- kmeans(pca_data, centers = 5, nstart = 10)

# Step 4 : Add cluster assignments to the PCA data
pca_data$Cluster <- as.factor(pca_kmeans$cluster)

#  Step 5 : Plot the PCA clusters
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "PCA Clusters (k = 5)",
       x = "Principal Component 1 (PC1)",
       y = "Principal Component 2 (PC2)",
       color = "Cluster") +
  theme_minimal()


