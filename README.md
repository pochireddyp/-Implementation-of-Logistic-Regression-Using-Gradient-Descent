# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Visulaize the data and define the sigmoid function, cost function and gradient descent.

4.Import linear regression from sklearn.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Obtain the graph.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: POCHIREDDY.P
RegisterNumber:  212223240115
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
## DATASET
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/798c5010-a00a-4ffd-90e9-74364362f199)

## datatypes
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/6c1d95b5-ea7a-40c1-b34b-2b9939545c27)
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/b1fac827-5527-4b73-8a1f-b7ff38d0a66a)

## Accuracy
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/34a90e6c-8bed-495c-a479-a73bdcdf0383)

## Array values of Y prediction
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/38d8e0ff-9b3f-4c69-bb61-c389781e800e)

## Array values of Y
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/557d3a50-7b04-4a3a-9f00-b5dfcfcdc07b)

## predicting with different values
![image](https://github.com/pochireddyp/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232043/a174cbfb-1f4f-4000-b5e1-407886a052ac)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

