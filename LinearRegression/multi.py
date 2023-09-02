from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from sklearn import datasets,metrics,linear_model
import seaborn as sbn

# load the boston dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
data=pn.read_csv(data_url,skiprows=22,header=None,sep="\s+")
x=np.hstack([data.values[::2,:],data.values[1::2,:2]])
y=data.values[1::2,2]

# this is to split x and y into train and test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

# this is to create linear regression
reg=linear_model.LinearRegression()

# let train model using training sets
reg.fit(x_train,y_train)

# let print the regression coeficient
print("The coefficients are \n ",reg.coef_)

# let me  the score where 1 means 100% perfect predictions
print('The variance score is \n ',reg.score(x_test,y_test))

# let me plot the residual errors

# let me set the plotting styles
# plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

# let me plot resual errors with scatter
plt.scatter(reg.predict(x_train),
            reg.predict(x_train)-y_train,
            color='indigo',label='Train data with python',s=10)
plt.title('This is the scatter of resual errors')
plt.xlabel('Error in unitless')
plt.ylabel('Error in unitless')

# this is plotting the line for zero error 
plt.hlines(xmin=1,xmax=100,linewidth=2,y=1, colors='indigo')
plt.vlines(ymin=-100,ymax=100,x=50, colors='red')

# this is ploting legend
plt.legend(loc='upper right')
plt.show()