#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:43:58 2017

@author: rhubner
"""
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import sys

from keras.models import Sequential
from keras.layers import Dense

language_data = load_files(container_path='/home/rhubner/scikit_learn_data/HACKATHON_TEXT-FILES', 
                          load_content=True,
                          encoding='UTF-8',
                          shuffle=True)

X_train, X_test, Y_train, Y_test = train_test_split(language_data.data, language_data.target, 
                                                    test_size=0.33, random_state=random.randint(1,4294967295))


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=5, batch_size=32)


loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
print("neural network stats")
print(loss_and_metrics)