# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:42:21 2017

@author: Kunal
"""
# CNN Practical Training

#part 1: - building the CNN network

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initailising the CNN

classifier = Sequential()

# Step 1 : Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), 
                             activation = 'relu'))

#step2 : Pooling

classifier.add(MaxPooling2D(pool_size = (2 , 2)))

#Step3 : flatteening

classifier.add(Flatten())

#Step4 : Full connection

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

''' In the training set we need to change the path of the directory and 
target_size will be correspond to the size given during convolution and same
technique is provided to the test_set'''
 
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

''' we need to change the variable name in model.fit as we have change the
name of the training and test set'''

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000/32)




