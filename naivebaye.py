import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error   

'''
We can use Scikit learn to build a simple linear regression model
you can use it use it like 
model = GaussianNB()
'''

## Step 1: Load Data from CSV File ####
dataframe = pd.read_csv('titanic.csv')
dataframe = dataframe.drop(["Name"] , axis = 1)

## Step 2: Plot the Data ####

ages = dataframe['Age'].values
fare = dataframe['Fare'].values
survived = dataframe['Survived'].values
color = []

for item in survived:
	if item == 0:
		color.append('red')
	else:
		color.append('green')

#plt.scatter(ages,fare , s =40 , color = color)
#plt.show()

## Step 3: Build a NB Model ####

feature = dataframe.drop(['Survived'] , axis = 1).values
target = dataframe["Survived"].values

feature_train , target_train = feature[:710] , target[:710]
feature_test , target_test = feature[710:] , target[710:]

model = GaussianNB()
model.fit(feature_train , target_train)
## Step 4: Print Predicted vs Actuals ####

predicted_values = model.predict(feature_test)

for item in zip(target_test , predicted_values):
	print("Actual was : " , item[0] , "Predicted was : " , item[1])

## Step 5: Estimate Error #### 

print (model.score(feature_test , target_test))