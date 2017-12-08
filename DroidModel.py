# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


language_data = load_files(container_path='/home/rhubner/scikit_learn_data/HACKATHON_TEXT-FILES', 
                          load_content=True,
                          encoding='UTF-8',
                          shuffle=True)

X_train, X_test, Y_train, Y_test = train_test_split(language_data.data, language_data.target, 
                                                    test_size=0.33, random_state=random.randint(1,4294967295))

from pprint import pprint

sample = 10

pprint(list(language_data.target_names))
print(Y_train[sample])
print(language_data.target_names[Y_train[sample]])
print("\n".join(X_train[sample].split("\n")[:7]))



# ----------    NB     ------------------------
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

text_clf = Pipeline([('vect', CountVectorizer(analyzer = 'word')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(X_train, Y_train)


predicted = text_clf.predict(X_test)
print("accuracy for NB : ")
print(np.mean(predicted == Y_test))



# ----------     SVM  -------------
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3, max_iter=5, random_state=42)),
])

_ = text_clf_svm.fit(X_train, Y_train)

predicted_svm = text_clf_svm.predict(X_test)
print("Support Vector Machines (SVM): ")
print(np.mean(predicted_svm == Y_test))
