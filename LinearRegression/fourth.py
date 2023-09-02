import matplotlib.pyplot as plt
import numpy as np

# this is the function to estimate the coeficient
def estimate_coef(x,y):
   
    # this is to find the the number of observation/points
    n=np.size(x)
   
    # this is the mean of x and y
    m_x=np.mean(x)
    m_y=np.mean(y)

    # let me find the sum of cross-deviations 
    ss_xy=np.sum(x*y)-n*m_y*m_x
    ss_xx=np.sum(x*x)-n*m_x*m_x
   
    # let me find slope and y_intercept means coeficients
    b_1=ss_xy/ss_xy    #this is slope
    b_0=m_y-b_1*m_x    #this is y_intercept
    return b_1,b_0


# this is the function to plot the regression line
def plotRegressionLine(x,y,b):
   
    # this is to plot the actual points as scatter
    plt.scatter(x,y,color='indigo')

    # this is predicted response vector
    y_pred=b[0]+b[1]*x

    # this is to plotting regression line
    plt.plot(x,y_pred,color='indigo')
    
    #this is  to put labels 
    plt.ylabel('This is the speed of the car in meter/second(m/s) ')
    plt.xlabel('This is the time used by the car in seconds(s)')
    plt.show()


# this is to define the main function 
def main():

    # These are data to be used 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # this is to estimate the coef by invoking the function
    b=estimate_coef(x,y)
    print(" Estimated coeficients are : \n b_o={} \
          \n b_1={} ".format(b[0],b[1]))

    # let me plot the regression line by invoking the function 
    plotRegressionLine(x,y,b)

#This is to check if the function name is main 
if __name__ == "__main__":
    main()


