from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from sklearn import datasets,metrics,linear_model

# load the boston dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
data=pn.read_csv(data_url,skiprows=22,header=None,sep="\s+")
x=np.hstack([data.values[::2,:],data.values[1::2,:2]])
y=data.values[1::2,2]

# this is to split x and y into train and test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

# this is to create linear regression
reg=linear_model.LinearRegression()