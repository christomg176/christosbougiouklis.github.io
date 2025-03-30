from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




X,y = load_iris(return_X_y=True)

X = X[:, :2]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test,y_train ,y_test = train_test_split(X,y, random_state=42, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print(accuracy_score(y_test,y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.title("KNN - Training Data (2 Features)")
plt.grid(True)
plt.show()