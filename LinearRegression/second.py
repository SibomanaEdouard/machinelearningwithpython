import pandas as pn
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# let me open the csv file with pandas model
trainData=pn.read_csv('../dataset/train.csv')
# let me read the components of csv file 
x=trainData['x'].tolist()
y=trainData['y'].tolist()


# # let add constant
# x=sm.add_constant(x)

# # this is to fit the model
# result=sm.OLS(y,x).fit()
# # this is to print the summary of the table
# print(result.summary())

plt.scatter(x,y)
plt.show()
# let me find the maximum x  and minimum
max_x=trainData['x'].max()
min_x=trainData['x'].min()
print(max_x)
print(min_x)
# let me arrange x
x=np.arange(min_x,max_x)

# this is to find the values of y
y=1.0143*x-0.4618
print(x)
print(y)
