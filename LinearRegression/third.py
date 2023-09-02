from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,metrics,linear_model
import pandas as pd


# this is csv file
data_url="http://lib.stat.cmu.edu/datasets/boston"
# let me read the file
data= pd.read_csv(data_url, sep="\s+",
                     skiprows=22, header=None)
x=np.hstack([data.values[::2,:],data.values[1::2,:2]])
y=data.values[1:2,2]
# print(x)
print(y)
# print(data)