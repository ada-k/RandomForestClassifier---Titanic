#!/usr/bin/env python
# coding: utf-8

# In[52]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[53]:


#importing datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()


# In[54]:


train.info()
train.describe()


# In[55]:


test.info()
test.describe()


# ## Null Values

# In[56]:


#dealing with null values in the train dataset
null = train.isna().sum().sort_values(ascending = False)
null_2 = test.isna().sum().sort_values(ascending = False)
null_values = pd.concat([null, null_2], keys = ['train null', 'test null'], axis = 1)
null_values.head()


# In[57]:


#replace all missing values in the age column w/the mean.
x = train.iloc[:, -7].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
train.iloc[:, -7] = x
train.isnull().sum()


# In[58]:


#replace all missing values in the age column w/the mean.(test)
x = test.iloc[:, -7].values
x = x.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x)
x = imputer.transform(x)
test.iloc[:, -7] = x
test.isnull().sum()


# In[59]:


#replace all missing values in the Embarked column w/the value with highest frequency.
y = train.iloc[:, -1].values
y = y.reshape(-1,1)
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(y)
y = imputer.transform(y)
train.iloc[:, -1] = y
train.isnull().sum()


# In[60]:


#dealing with the null fare value (test)
#Check the row with null Fare Value
test[test["Fare"].isnull()]


# In[61]:


#check for Fare prices for passengers in 3rd class who embarked from Southampton.
for a,b,c in zip(test["Fare"], test["Pclass"], test["Embarked"]):
    if b == 3 and c == "S":
        fare = a

#find the average fare for these passengers
class_3_list = [fare]
from statistics import mean
m = mean(class_3_list)

#replace the nan with this average
test["Fare"].fillna(m, inplace = True) 
test.isnull().sum()


# In[62]:


#dealing w/missing cabin data in both datasets
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train, test]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    


# ## dropping less significant columns

# In[63]:


#drop the cabin variable
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)


# ## Remapping categorical data to int type.

# In[64]:


#remapping female/male to 0/1 in sex column 
#remapping Embarked( C, Q S) to 0/1/2 in sex column
dataset = [train, test]
for data in dataset:
    data.Sex[data.Sex == 'female'] = 0
    data.Sex[data.Sex == 'male'] = 1
    data.Embarked[data.Embarked == 'C'] = 0
    data.Embarked[data.Embarked == 'Q'] = 1
    data.Embarked[data.Embarked == 'S'] = 2


# ## Exploring variables relation to survival

# In[65]:


#age
survived = 'survived'
not_survived = 'not survived'
ax = sns.distplot(train[train['Survived']==1].Age.dropna(), bins=18, label = survived,kde =False)
ax = sns.distplot(train[train['Survived']==0].Age.dropna(), bins=40, label = not_survived,kde =False)
ax.legend()


# In[66]:


#sibsp
sns.barplot(x='SibSp', y='Survived', data=train)


# In[67]:


#parch
sns.barplot(x='Parch', y='Survived', data=train)


# In[68]:


#fare
survived = 'survived'
not_survived = 'not survived'
ax = sns.distplot(train[train['Survived']==1].Fare.dropna(), bins=18, label = survived,kde =False)
ax = sns.distplot(train[train['Survived']==0].Fare.dropna(), bins=40, label = not_survived,kde =False)
ax.legend()


# ## creating categories

# In[69]:


#creating categories
#age
data = [train, test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

#sibsp
data = [train, test]
for dataset in data:
    dataset.loc[ dataset['SibSp'] == 0, 'SibSp'] = 0
    dataset.loc[(dataset['SibSp'] > 0) & (dataset['SibSp'] <= 2), 'SibSp'] = 1
    dataset.loc[(dataset['SibSp'] > 2) & (dataset['SibSp'] <= 4), 'SibSp']   = 2
    dataset.loc[ dataset['SibSp'] > 4, 'SibSp'] = 3
    dataset['SibSp'] = dataset['SibSp'].astype(int)
    
#parch
data = [train, test]
for dataset in data:
    dataset.loc[ dataset['Parch'] <= 3, 'Parch'] = 0
    dataset.loc[(dataset['Parch'] > 3) & (dataset['Parch'] <= 4), 'Parch'] = 1
    dataset.loc[(dataset['Parch'] > 4) & (dataset['Parch'] <= 5), 'Parch']   = 2
    dataset.loc[(dataset['Parch'] > 5) & (dataset['Parch'] <= 6), 'Parch']   = 3
    dataset['Parch'] = dataset['Parch'].astype(int)
    
#fare
data = [train, test]
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 200), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 200, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[70]:


#handling the name column
data = [train, test]
titles = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)


# In[71]:


data = [train, test]
for i in data:
    print(i.describe())


# ## Fitting RandomForest Classifier

# In[86]:


data = [train, test]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']


# ## Submission File

# In[89]:


# Fitting Random Forest Classifier
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()

# Fitting Random Forest Classification to Training set
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest



# In[90]:


print(Y_prediction)


# In[91]:


new_id = test.iloc[:, 0]
file = pd.DataFrame({'PassengerId':new_id, 'Survived': predicted_survived})
submission = file.set_index('PassengerId')
submission.to_csv('submission_2.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




