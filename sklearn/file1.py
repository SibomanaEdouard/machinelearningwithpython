from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# let me load dataset called iris
dataset=datasets.load_iris()
# print(dataset)
# let fit the data to the model
model=DecisionTreeClassifier()
model.fit(dataset.data,dataset.target)
print(model)

# this is to make prediction
expected=dataset.target
predicted=model.predict(dataset.data)

# this is to summarize model
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))