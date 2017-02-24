import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


data = []
clf = svm.SVC(kernel = "rbf")

# returns all data
def loadAllData():
	data = pandas.read_csv('C:\\Users\\sawye\\Desktop\\dmFinal\\creditcard.csv')
	return data

# returns the first num lines of data
def loadNData(num):
	data = pandas.read_csv('C:\\Users\\sawye\\Desktop\\dmFinal\\creditcard.csv', nrows = num)
	return data

# load all 492 frauds
def loadAllFraud():
	data = loadAllData()
	numFraud = data.Class.sum()
	data = data.sort_values(['Class'], ascending = False)
	return data.head(numFraud)

#load all non fraudulent data
def loadAllTrue():
	data = loadAllData()
	numTrue = len(data.index) - data.Class.sum()
	#print(numTrue)
	data = data.sort_values('Class', ascending = False)
	allTrues = data.tail(numTrue)
	print("the non frauds returned this many frauds: ",allTrues.Class.sum())
	return allTrues




Frauds = loadAllFraud()
Trues = loadAllTrue()

Data = loadNData(10000)

Data = pandas.concat([Frauds * 10, Data])
print(Data.Class.sum())
Target = Data.Class

X_train, X_test, y_train, y_test = train_test_split(Data, Data.Class, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)

'''
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, Data, Target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))         
'''


clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
print("The accuracy was",clf.score(X_test, y_test) )



X = Data
y = Target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(clf, X, y, cv=2)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


#print("Num of frauds in train: ", y_train.sum())
#print("Num of frauds in test: ", y_test.sum())
#printNLines(10)
#dataTrain = loadNData(1000)
#print(data)
#myY = dataTrain.Class
#print(myY)
#clf.fit(dataTrain, myY)
#print(clf)
#print(dataTrain.shape)
#dataTest = l
