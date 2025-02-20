#a
# Step 1: Load the mtcars dataset
data(mtcars)

# Step 2: Standardize the data (z-score)
mtcars_standardized <- scale(mtcars)

# Step 3: Compute the pairwise dissimilarity matrix using Euclidean distance
dissimilarity_matrix <- dist(mtcars_standardized, method = "euclidean")

# Convert the dissimilarity matrix to a regular matrix for easier indexing
dissimilarity_matrix_1 <- as.matrix(dissimilarity_matrix)

# Step 4: Extract the dissimilarity between Mazda RX4 and all Merc 450 cars
# Identify the row names for Mazda RX4 and Merc 450 cars
mazda_rx4 <- "Mazda RX4"
merc_450_cars <- rownames(mtcars)[grepl("Merc 450", rownames(mtcars))]

# Extract the dissimilarity values
dissimilarity_values <- dissimilarity_matrix_1[mazda_rx4, merc_450_cars]

# Step 5: Report the results in a table
result_table <- data.frame(
   
  Mazda_RX4 = round(dissimilarity_values, 3)
)

print("Dissimilarity between Mazda RX4 and Merc 450 cars:")
print(result_table)

#b
# Step 1: Perform hierarchical clustering with single linkage
single_linkage <- hclust(dist(mtcars_standardized), method = "single")

# Step 2: Perform hierarchical clustering with complete linkage
complete_linkage <- hclust(dist(mtcars_standardized), method = "complete")


# Step 3: Plot single linkage dendrogram
plot(single_linkage, main = "Dendrogram (Single Linkage)", xlab = "", sub = "", cex = 0.9)
rect.hclust(single_linkage, k = 3, border = 2:4)  # Highlight 3 clusters

# Step 4: Plot complete linkage dendrogram
plot(complete_linkage, main = "Dendrogram (Complete Linkage)", xlab = "", sub = "", cex = 0.9)
rect.hclust(complete_linkage, k = 3, border = 2:4)  # Highlight 3 clusters

#c
# Step 1: Cut the single linkage dendrogram into 4 clusters
single_clusters <- cutree(single_linkage, k = 4)

# Step 2: Cut the complete linkage dendrogram into 4 clusters
complete_clusters <- cutree(complete_linkage, k = 4)

# Step 3: Display the number of observations in each cluster
# For single linkage
single_cluster_counts <- table(single_clusters)
print("Number of observations in each cluster (Single Linkage):")
print(single_cluster_counts)

# For complete linkage
complete_cluster_counts <- table(complete_clusters)
print("Number of observations in each cluster (Complete Linkage):")
print(complete_cluster_counts)


#d
# Step 1: Add cluster assignments to the original dataset
mtcars$single_cluster <- single_clusters
mtcars$complete_cluster <- complete_clusters

# Step 2: Combine Cluster 1 from single and complete linkage
combined_cluster_1 <- mtcars[mtcars$single_cluster == 1 | mtcars$complete_cluster == 1, c("cyl", "hp")]

# Step 3: Calculate the combined means for Cluster 1
combined_cluster_1_means <- colMeans(combined_cluster_1)

# Step 4: Combine Cluster 3 from single and complete linkage
combined_cluster_3 <- mtcars[mtcars$single_cluster == 3 | mtcars$complete_cluster == 3, c("cyl", "hp")]

# Step 5: Calculate the combined means for Cluster 3
combined_cluster_3_means <- colMeans(combined_cluster_3)

# Step 6: Report the results
print("Combined means of 'cyl' and 'hp' for Cluster 1 (Single and Complete Linkage):")
print(combined_cluster_1_means)

print("Combined means of 'cyl' and 'hp' for Cluster 3 (Single and Complete Linkage):")
print(combined_cluster_3_means)

