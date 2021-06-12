#importing required libraries
import pandas as pd
from sklearn import model_selection,neighbors,preprocessing
import numpy as np


#reading the dataset
df=pd.read_csv('car.data')

#preprocessing the data
#converting string values of features to numerical values
df=df.replace({'buying_price':{'vhigh':4,'high':3,'med':2,'low':1}})
df=df.replace({'maint_price':{'vhigh':4,'high':3,'med':2,'low':1}})
df=df.replace({'doors':{'5more':5}})
df=df.replace({'person_cap':{'more':6}})
df=df.replace({'lug_boot':{'small':1,'med':2,'big':3}})
df=df.replace({'safety':{'low':1,'med':2,'high':3}})
df=df.replace({'quality':{'unacc':1,'acc':2,'good':3,'vgood':4}})


#assigning the features and labels
x=np.array(df.drop(['quality'],1))
y=np.array(df['quality'])

#splitting the dataset to test and train set
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

#initialising KNN algorithm
clf=neighbors.KNeighborsClassifier(n_neighbors=9)

#training the model
clf.fit(x_train,y_train)

accuracy=clf.score(x_test,y_test)
predict_values=[]
l=[]
text=['Enter the buying_price\n1. 1 for low\n2. 2 for medium\n3. 3 for high\n4. 4 for very high','Enter the cost of maintenance:\n1. 1 for low\n2. 2 for medium\n3. 3 for High\n4. 4 for Very high','Enter the number of doors of the Car\nEnter 5 for cars with 5 or more doors','Enter the seating Capacity:\nEnter 5 for greater than or equal to 5','Enter the luggage capacity:\n1. 1 for Low\n2. 2 for Medium\n3. 3 for Large','Enter the Safety:\n1. 1 for Low\n2. 2 for Medium\n3. 3 for High']
for i in range(len(text)):
    print(text[i])
    d=int(input())
    l.append(d)
predict_values.append(l)

#predicting the user input
prediction=clf.predict(predict_values)


final_predict=prediction[0]-1
text_quality=['Very Poor','Average','Good','Excellent']

#predicted value
print('The quality of the car:',text_quality[final_predict])








