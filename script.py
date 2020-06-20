import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")
print(df.head())

#Total honey production per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
#Create X and reshape
X = prod_per_year['year']
X = X.values.reshape(-1, 1)
#Create y
y = prod_per_year['totalprod']

#Create and Fit a Linear Regression Model
regr = LinearRegression()
regr.fit(X, y)
print(regr.coef_[0])
print(regr.intercept_)
#Predicting y
y_predict = regr.predict(X)

#Plotting a scatter plot
plt.scatter(X, y)
plt.plot(X, y_predict)
plt.show()
#it looks like the production of honey has been in decline, according to this linear model.

#Predicting what the year 2050 might look like in terms of honey production.
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)
#You can think of reshape() as rotating this array. Rather than one big row of numbers, X_future is now a big column of numbers — there’s one number in each row.

future_predict = regr.predict(X_future)
#Plot
plt.plot(X_future, future_predict)
plt.show()
#According to the model, by 2050, honey production will be less than 1000000.
