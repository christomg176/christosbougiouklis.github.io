from sklearn.linear_model import  LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import r2_score,mean_squared_error


X = np.random.rand(1000,1)
y = 2 * X + np.random.rand(1000,1)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, random_state=42, test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(model.coef_,model.intercept_)
print(r2_score(y_test,y_pred), mean_squared_error(y_test,y_pred))
plt.scatter(X_train,y_train)
plt.plot(X_test,y_pred, color = "red")
plt.show()
print(model.coef_,model.intercept_)
print(r2_score(y_test,y_pred), mean_squared_error(y_test,y_pred))