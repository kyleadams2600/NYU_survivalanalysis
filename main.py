import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#a simple linear regression using UCI iris

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

print(df.head())


x = df.iloc[:, 2].values.reshape(-1, 1)
y = df.iloc[:, 3].values.reshape(-1, 1)
linear_regressor = LinearRegression().fit(x, y)
y_pred = linear_regressor.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='green')
plt.title("Predicting petal width from length")
plt.show()

#machine learning style


#Split the data into training/testing sets
x_train = x[:-75]
x_test = x[-75:]

#Split the targets into training/testing sets
y_train = y[:-75]
y_test = y[-75:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.title("Predicting Petal Width from Length")

plt.show()
