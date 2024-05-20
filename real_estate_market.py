# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:44:12 2024

@author: bahadir sahin
"""

########### Multiple Linear Regression ###########

# necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline


# Downloading the dataset and displaying the first five rows:
    
#veri = pd.read_excel(“XL_test.xlsx”, engine='openpyxl')
veri = pd.read_excel("https://www.dropbox.com/s/luoopt5biecb04g/SATILIK_EV1.xlsx?dl=1")
print("\n first 5 line")
print(veri.head())

# Let's look at the size of the dataset: 
print("\n data shape")
print(veri.shape)


# Let's look at the summary of descriptive statistics of the dataset:
print("\n data describe")
print(veri.describe())


# Let's check if there is any missing data in the dataset:
print("\n data is null?")
print(veri.isnull().any())


# Visualization of the correlations between all variables in the dataset:
print("\n coorelation visualization")
plt.figure(figsize=(12,10))
cor = veri.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#the scatter plots of all variables with each other:
# print("\n scatter plots visualization")
# sns.pairplot(veri)

# Let's look at the distribution of the target variable 'Fiyat'
# print("\n visualization target variable (fiyat)")
# sns.histplot(veri['Fiyat'])
# plt.show()


# Define the target (Y) and feature variables (X):
X = veri[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
Y = veri['Fiyat']


# randomly split the dataset into training and test sets. To allocate 80% of the dataset to training and 20% to the test set, we enter a value for the "test_size" parameter. Here test_size=0.2 means 80% of the dataset is training and 20% is the test set. The "random_state" parameter ensures that the split is random each time.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create a model using the training data:
model = LinearRegression()
model.fit(X_train, y_train)

# look at the intercept and coefficients for the feature variables in the training dataset:
print("\n model intercept=",model.intercept_)
print("\n model coeff=", model.coef_)


coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Feature_Coefficients'])
print("\ncoeff_df=" , coeff_df)

# compare the predicted values with the actual values for the test dataset:

print("\ncompare the predicted values with the actual values for the test dataset")
y_pred_eğitim = model.predict(X_train)
for i, prediction in enumerate(y_pred_eğitim):
    print('predicted house price: %s, actual house price: %s' % (prediction, y_train.iloc[i]))

print("\n ")

#the r^2 for the target variable (y_train) in the training dataset:
from sklearn.metrics import r2_score
print("\nr2_score for train set=", r2_score(y_train, y_pred_eğitim))

# Eğitim veri setindeki hedef değişkeninin gerçek ve tahmin edilmiş değerlerinin serpilme grafiği:
plt.scatter(y_train, y_pred_eğitim)
plt.show()

# compare the predicted values with the actual values for the test dataset:
print("\ncompare the predicted values with the actual values for the test dataset:")
y_pred_test = model.predict(X_test)
for i, prediction in enumerate(y_pred_test):
    print('predicted house prices: %0.2f, actual house price: %s' % (prediction, y_test.iloc[i]))

#the r^2 for the target variable (y_test) in the test dataset:
print("\nr2_score for test set=",r2_score(y_test, y_pred_test))

# Scatter plot of the actual and predicted values for the target variable in the test dataset:
print("\nscatter plot of actual and predicted values for target value visualization")
plt.scatter(y_test, y_pred_test)
plt.show()

#predict the price of a house with 3 rooms, 8 years old, on the 4th floor of a building, with a net usable area of 105 m2:
Oda_Sayısı = 3
Net_m2 = 105
Katı = 4
Yaşı = 8

new_house = pd.DataFrame([[Oda_Sayısı, Net_m2, Katı, Yaşı]], columns=['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı'])
print('\nNew House Price (₺):', model.predict(new_house))

