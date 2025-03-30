from cProfile import label

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt


X = np.random.rand(1000).reshape(-1,1)
y = 4 * X + np.random.rand(1000).reshape(-1,1)

X_train , X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

model =LinearRegression()
model.fit(X_test,y_test)

y_pred = model.predict(X_test)

print(model.coef_,model.intercept_)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(mse,r2)

plt.scatter(X_train,y_train)
plt.plot(X_test,y_pred, color = "red",label = "prediciton")
plt.legend()
plt.show()