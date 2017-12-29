
# Date preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#encoding categorical data which are country and gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[ :, 1] = labelencoder_X_1.fit_transform(X[ :, 1])
labelencoder_X_2 = LabelEncoder()
X[ :, 2] = labelencoder_X_2.fit_transform(X[ :, 2])
#Creating dummy variable so that the string not behvaves greater to other string

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



'''
# Fitting classifier to the Training set
# Create your classifier here

#part 2 lets create a ANN network
# import the keras lib and necessary packages

import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout # Dropout Techinque

#initializing the Neural Network
classifier = Sequential ()

#Adding the input and first hidden layers

classifier.add(Dense(units =6, input_dim= 11 , kernel_initializer= 'uniform', 
                     activation= 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layers

classifier.add(Dense(units =6, kernel_initializer= 'uniform', 
                     activation= 'relu'))
classifier.add(Dropout(p = 0.1))
# Adding the output layer


classifier.add(Dense(units =1, kernel_initializer= 'uniform', 
                     activation= 'sigmoid'))

# Compling the ANN

classifier.compile (optimizer = 'adam', loss = 'binary_crossentropy' , 
                    metrics = ['accuracy'])

# Fitting the ANN to the training set

classifier.fit (X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


#predicting the example set to the above model
""" Following example is tested by the model
Geography = france
credit score = 600
gender = male
age = 40
tenure = 3
balance = 60000
No of products = 2
has credit card = yes
Is Active meber = yes
Estmated salary = 50000 """

#So creating object for the above example
new_predict = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1
                                                         ,1 ,50000]])))
new_predict = (new_predict > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# part 4- Evaluating the ANN, improving and tunning it

#Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
 

def Build_classifier():
    classifier = Sequential ()
    classifier.add(Dense(units =6, input_dim= 11 , kernel_initializer= 'uniform', 
                     activation= 'relu'))
    classifier.add(Dense(units =6, kernel_initializer= 'uniform', 
                     activation= 'relu'))
    classifier.add(Dense(units =1, kernel_initializer= 'uniform', 
                     activation= 'sigmoid'))
    classifier.compile (optimizer = 'adam', loss = 'binary_crossentropy' , 
                    metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,
                             cv = 10, n_jobs = 1)
mean = accuracies.mean()
Varirance = accuracies.std()

'''



#Parameter Tunning

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
 

def Build_classifier(optimizer):
    classifier = Sequential ()
    classifier.add(Dense(units =6, input_dim= 11,kernel_initializer= 'uniform', 
                     activation= 'relu'))
   # classifier.add(Dropout(p = 0.2))

    classifier.add(Dense(units =6, kernel_initializer= 'uniform', 
                     activation= 'relu'))
   # classifier.add(Dropout(p = 0.2))

    classifier.add(Dense(units =1, kernel_initializer= 'uniform', 
                     activation= 'sigmoid'))
    classifier.compile (optimizer = optimizer, loss = 'binary_crossentropy' , 
                    metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Build_classifier)
parameters = {'batch_size' : [25,32],
               'epochs': [100,500],
               'optimizer' :['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

    