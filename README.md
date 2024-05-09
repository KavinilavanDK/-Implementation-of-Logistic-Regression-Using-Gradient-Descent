# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for sigmoid, loss, gradient and predict and perform operations.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KAVI NILAVAN DK
RegisterNumber:  212223230103
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)
```
## Output:
#### Read the file and display:
![ml 1](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/1be2aaa3-31e4-4ace-be2f-5a67b81016a7)
#### Categorizing columns:
![ml 2](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/f5a11517-93db-454d-ae79-2741d48f41c5)
#### Labelling columns and displaying dataset:
![ml 3](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/6a3cbcb9-ba0a-41b4-b5bf-502ac99bf948)
#### Display dependent variable:
![ml 4](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/322946a3-9b9a-4b15-983e-01d8dd138471)
#### Printing accuracy:
![ml 5](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/a94f17d4-6aca-442e-87ba-edf3e6b72095)
#### Printing Y:
![ml 6](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/3063101f-2980-4e7c-aaf3-f47f5c1f0399)
#### Printing y_prednew:
![ml 7](https://github.com/KavinilavanDK/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870429/c32b3334-7098-4031-b821-1fb05f850c4f)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

