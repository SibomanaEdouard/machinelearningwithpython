# let me learn about the import 

# import math as mt
# print(mt.pi)
# print(type(mt))
# print(dir(mt))

# this module is used to deal with arrays 
import numpy as np 

# let me create the array
x=np.array([[2.6,3,5],[2,3,6],[12,34,56],[10,20,40]])
y=np.array([[2.6,3,5],[2,3,6],[12,34,56],[10,20,40]])
print('This is original array : ',x)
# y=np.array([[4,3],[4,2]])
print("This is sliced array \n ", x[::3])
print("I added 3 to each \n",x+3)
print("The sum of all elements in the array : ",np.sum(x))
print("The data type  is ",x.dtype)
print("The type is ",type(x))
print("The sum of two arrays : \n",np.add(x,y))
print("The square root of the array is \n",np.sqrt(x))
print("The transporse of the array is \n",x.T)
