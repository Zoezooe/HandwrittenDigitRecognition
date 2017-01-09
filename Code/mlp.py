# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 10:49:32 2016
Reference: http://machinelearningmastery.com/
@author: Siaki
"""

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

seed=4	#Random number set to some value.
numpy.random.seed(seed)

#loading the datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
no_of_pixels = X_train.shape[1] * X_train.shape[2]

#Processing 784 dimensional vectors to flatten the data
X_train = X_train.reshape(X_train.shape[0], no_of_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], no_of_pixels).astype('float32')

#Normalizing the Data samples
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def mlp():
    model = Sequential()
    model.add(Dense(no_of_pixels, input_dim=no_of_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

mlpmodel = mlp()
mlpmodel.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
scores = mlpmodel.evaluate(X_test, y_test, verbose=0)

print(" Error: %.2f%%" % (100-scores[1]*100))