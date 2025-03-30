from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X,y =load_iris(return_X_y=True)


model = DecisionTreeClassifier()

X_train ,X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

model.fit(X_train,y_train)

pred_y = model.predict(X_test)

plt.figure(figsize=(10,10))
plot_tree(model,max_depth=10, feature_names=iris.feature_names, class_names=iris.target_names, filled= True)
plt.show()

