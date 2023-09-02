import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.axes as axe

# let me read csv file
data=pd.read_csv('../dataset/data_for_lr.csv')
# this is to drop the missing values
data=data.dropna()

# let me train the dataset and labels
train_input=np.array(data.x[0:500]).reshape(500,1)
train_output=np.array(data.y[0:500]).reshape(500,1)

# let me validate the dataset
test_input=np.array(data.x[500:700]).reshape(199,1)
test_output=np.array(data.y[500:700]).reshape(199,1)

class LinearRegression:
    def __init__(self):
     self.parameters={}
    
    def forword_propagation(self,train_input):
       m=self.parameters['m']
       c=self.parameters['c']
       predictions=(m*train_input)+c
       return predictions 
    
    def cost_function(self,predictions,train_input):
        cost=np.mean((train_input-predictions)**2)
        return cost
    
    def backword_propagation(self,train_input,train_output,predictions):
       derivatives={}
       df=(train_input-predictions)* -1
       dm=np.mean(np.multiply(train_input,df))
       dc=np.mean(df)
       derivatives['dm']=dm
       derivatives['dc']=dc
       return derivatives
    
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']
 
    def train(self, train_input, train_output, learning_rate, iters):
        #initialize random parameters
        self.parameters['m'] = np.random.uniform(0,1) * -1
        self.parameters['c'] = np.random.uniform(0,1) * -1
         
        #initialize loss
        self.loss = []
         
        #iterate
        for i in range(iters):
            #forward propagation
            predictions = self.forword_propagation(train_input)
 
            #cost function
            cost = self.cost_function(predictions, train_output)
 
            #append loss and print
            self.loss.append(cost)
            print("Iteration = {}, Loss = {}".format(i+1, cost))
 
            #back propagation
            derivatives = self.backword_propagation(train_input, train_output, predictions)
 
            #update parameters
            self.update_parameters(derivatives, learning_rate)
 
        return self.parameters, self.loss
    #This is predictions on test data
# y_pred=test_input*parameters['m']
#     #Example usage
# linear_reg = LinearRegression()
# parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)
# plt.plot(test_input,test_output,'+',label='')
#Prediction on test data

# y_pred = test_input*parameters['m'] + parameters['c']

# Plot the regression line with actual data pointa
plt.plot(test_input, test_output, '+', label='Actual values')
# plt.plot(test_input, y_pred, label='Predicted values')
plt.xlabel('Test input')
plt.ylabel('Test Output or Predicted output')
plt.legend()
plt.show()






