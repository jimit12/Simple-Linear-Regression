# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:17:53 2020

@author: user
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
regressor.fit(X_train, y_train)
X_test=X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()