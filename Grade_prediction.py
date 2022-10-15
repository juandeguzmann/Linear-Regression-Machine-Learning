import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplt
from matplotlib import style
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']] #all attributes

predict = 'G3' #determining what G3 is, from testing, this is called a label - what you are looking for

X = np.array(data.drop([predict], 1)) #return new dataframe, removing G3
y = np.array(data[predict]) #all of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
best = 30

''''
#This trains the model 30 times to find th emost accurate one
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    #Making the best fit line, training the model and saving the model via pickle:
    linear = linear_model.LinearRegression() #stores the line of best fit here

    linear.fit(x_train, y_train) #using y and x train to find he best fit line
    acc = linear.score(x_test, y_test) #accuracy of the test
    print(acc)

    if acc > best:
        best = acc
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f) #saving pickle file in the directory'''

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

print('Co: \n', linear.coef_) # m coeffients for the linear line
print('Intercep: \n', linear.intercept_) # y intercept

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'absences'
style.use('ggplot')
pyplt.scatter(data[p], data['G3'])
pyplt.xlabel(p)
pyplt.ylabel('Final grade')
pyplt.show()