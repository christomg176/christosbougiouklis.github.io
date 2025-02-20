#a
# Step 1: Load the Groceries dataset
library(arules)
data("Groceries")

# Step 2: Summarize the dataset
summary(Groceries)

#b
# Step 1: Apply the Apriori algorithm with min support = 0.01
frequent_itemsets <- apriori(Groceries, 
                             parameter = list(supp = 0.01, target = "frequent itemsets"))


#Step 2: Sort the frequent itemsets by support in descending order
sorted_itemsets <- sort(frequent_itemsets, by = "support")

#Step 3: Inspect the method
inspect(sorted_itemsets[1:5])

#c
# Step 1:Generate association rules with a minimum support of 0.01 and minimum confidence of 0.5
rules <- apriori(Groceries, parameter = list(support = 0.01, confidence = 0.5, target = "rules"))

# Step 2 : Sort the rules by lift in descending order
sorted_rules <- sort(rules, by = "lift")

# Step 3: Inspect the top-5 rules
inspect(sorted_rules[1:5])

# Step 4: Plot the top-5 rules as a graph
plot(sorted_rules[1:5], method = "graph", engine = "htmlwidget")

