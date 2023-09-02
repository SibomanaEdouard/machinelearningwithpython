import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# import seaborn as sb
# iris=sb.load_dataset("iris")
# print(iris.head())
# iris.to_csv("iris.csv",index=False)

# this is to read csv file called dataset
df=pd.read_csv('iris.csv')

# this is to  split the data into features and label
x=df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df[['species']]

# this is to convert y into one dimension
y = y.values.ravel()
# this is to train  the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# this is svm model
model=SVC()
model.fit(x_train,y_train)

# let me test its accuracy
accuracy=model.score(x_test,y_test)
print("Accuracy  is " , accuracy)



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# # Read the CSV file called dataset
# df = pd.read_csv('iris.csv')

# # Split the data into features and labels
# x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# y = df['species']  # Extracting a 1D array (Series) for the target

# # Perform the train-test split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Create an SVM model
# model = SVC()
# model.fit(x_train, y_train)

# # Test the model's accuracy
# accuracy = model.score(x_test, y_test)
# print("Accuracy is", accuracy)
