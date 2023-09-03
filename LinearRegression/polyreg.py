import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

# this is to define the function called main

def main():
    # this is to create the dataset
        x= np.array( [ [1], [2], [3], [4], [5], [6], [7] ] )
        y = np.array( [ 45000, 50000, 60000, 80000, 110000, 150000, 200000 ] )
        # this is to train the model
        model=LinearRegression()
        model.fit(x,y)

        # this is to make model prediction
        y_pred=model.predict(x)

        # this is to set styles
        plt.style.use('dark_background')

        # this is to make visualization using graphs

        plt.scatter(x,y,color='indigo',label='Trained model')
        plt.xlabel('This is x-axis')
        plt.ylabel('This is y-axis')
        plt.title('THIS IS DATA VISUALIZATION')
        plt.legend(loc='upper right') 
        plt.plot(x,y_pred)
        plt.show()



# this is to invoke the main function 
if __name__ =='__main__':
        main()