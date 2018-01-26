#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:16:33 2018

@author: griffin
"""

# =============================================================================
# K-nearest neighbors the purpose of this algorithm is to take input data and
# cluster it into groups it is meant for classification of data
#
# the k is just a variable representing a number. telling you which data wighs
# in on your anylasis. for example if k=2 then you look at the 2 points closest
# to your point you're classifying. Typically you will be making k an odd number
# so there isn't a split in your confidence level. To find nearest neighbors we
# are going to be using euclidean distance. This method isn't the most efficient 
# for large data sets as it is computationally intensive. It is still pretty good 
# for the purpose of classification. 
# =============================================================================

import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,2,1,3,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)