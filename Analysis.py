# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:04:05 2020

@author: Himanshu
"""
#Importing usefull libraries
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt


#Reading the dataset
data = pd.read_csv('train.csv')
target = data.iloc[:,1]

#Removing Survived column from data. As it is our target.
data = data.drop(columns = 'Survived')
#Let's plot the pair plot of all the columns.
#It will give us the idea about the dataset.
pairplot = sns.pairplot(data, hue = "Survived") 


#Calculation of NA values in our dataset
for i in data.columns:
    print(i + '          ' + str(sum(pd.isna(data[i]))))

#There are hugh missing values in Cabin and Age column.
#Embarked have only 2 missing values.
#Let's do analysis of Embarked
from sklearn.cluster import KMeans

unique_places = data['Embarked'].unique()
value_count = data['Embarked'].value_counts() #more than half of the passengers were from S
len(data) - sum(value_count) #we can see that there are 2 NAN values

s_fare = data[data['Embarked'] == 'S'].Fare
c_fare = data[data['Embarked'] == 'C'].Fare
q_fare = data[data['Embarked'] == 'Q'].Fare

s_mean = sum(s_fare)/ len(s_fare)
c_mean = sum(c_fare)/ len(c_fare)
q_mean = sum(q_fare)/ len(q_fare)
sns.distplot(s_fare)
sns.distplot(c_fare)
sns.distplot(q_fare)
data['Embarked'] = data['Embarked'].replace({'S': 0, 'C' : 1, 'Q' : 2, np.nan : -2})
nan_fare = data[data['Embarked'] == -2].Fare #Both have fare of 80
nan_name = data[data['Embarked'] == -2].Name
nan_age = data[data['Embarked'] == -2].Age #38 and 62
nan_parch = data[data['Embarked'] == -2].Parch
nan_SibSp = data[data['Embarked'] == -2].SibSp #both of these were travelling alone.
nan_name = data[data['Embarked'] == -2].Cabin  #both of these were alloted the same cabin(Something interesting).
#B28
nan_data = data[data['Cabin'] == 'B28']
#we can see that there are only 2 entries of B28.
#and both of these have same ticket number.
#both of these could be friends.
#travelling in 1st class.




#We will try to figure out the missing value using k means clustering algorithm
#We will keep the No. of cluster to be 3 as there are three different places from where people
#boarded on titanic
kmeans = KMeans(n_clusters=3, random_state=0).fit(embarked)