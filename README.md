# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YASHWANTH RAJA DURAI
RegisterNumber: 212222040184

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

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

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```
## Output:
![Screenshot 2024-04-22 184513](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/424b7114-6771-47c9-b1a5-4bbb190d290c)

![Screenshot 2024-04-22 184533](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/c9209dec-3349-4630-9aa3-4407978cbe64)

![Screenshot 2024-04-22 184550](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/53be9363-1ee2-4215-9c7b-bee2d858ec9b)

![Screenshot 2024-04-22 184608](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/25c7e333-498e-4424-8863-67d018b5cf4b)

![Screenshot 2024-04-22 184626](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/0a2a07a9-02e9-485e-a1ef-60a379ca989b)

![Screenshot 2024-04-22 184646](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/781563d7-4f95-4a28-b036-6e92a66986c9)

![Screenshot 2024-04-22 184710](https://github.com/yashwanthrajadurai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128462316/12a2fb91-86c1-49ae-a24f-ba293dd0806c)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
