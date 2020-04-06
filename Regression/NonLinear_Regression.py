import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.metrics import r2_score
from sklearn import linear_model
from scipy.optimize import curve_fit
# https://github.com/MrinmoiHossain/Online-Courses-Learning/blob/master/Coursera/Machine%20Learning%20with%20Python-IBM/Week-2/Exercise/ML0101EN-Reg-NoneLinearRegression-py-v1.ipynb

df = pd.read_csv("china_gdp.csv")
# print(df.head(10))

# Plotting the dataset
# plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
# plt.plot(x_data, y_data, 'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()

# Building The Model¶
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Plot resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()