#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:21:48 2018

@author: griffin j. kennedy
"""

# =============================================================================
# Breaking Down Linear Regression. The Purpuse of this code is to
# understand how and why linear regression works. In generall L.R.
# is used to find the 'line of best fit' for a set of data points.
# refer to Gilbert Strang's Linear Algebra notes for the math involved.
#
# To get the most value out of L.R. first look at the data and see if there
# is a correlation and see if L.R. is possible and if it can add VALUE.  
# =============================================================================

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

#plt.scatter(xs,ys)
#plt.show()

def best_fit_line(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs * ys)) / 
         ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m, b

# =============================================================================
# Okay... this is great but how accurate is our best fit line?
# Well we will need to calculate the 'squared error'.
# error is the difference between the point and the regression line.
# =============================================================================

def squared_error(ys_data, ys_regline):
    return sum((ys_regline - ys_data)**2)

def coefficient_of_determination(ys_data, ys_regline):
    y_mean_line = [mean(ys_data) for y in ys_data]
    squarred_error_regline = squared_error(ys_data, ys_regline)
    squarred_error_ymean = squared_error(ys_data, y_mean_line)
    return 1 - (squarred_error_regline / squarred_error_ymean)

# =============================================================================
# How do we make sure that our code is working how we would expect? Yes our code runs
# but is it working properly? To test this let's use sample data.
# =============================================================================

def create_dataset(hm, varience, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-varience, varience)
        ys.append(y)
        if correlation and correlation == 'positive':
            val += step
        elif correlation and correlation == 'negative':
            val -= step
    xs = [i for i in range(len(ys))]
      
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)


# =============================================================================
# Putting it all together
# =============================================================================

xs, ys = create_dataset(40, 40, 2, correlation= 'negative')

m, b = best_fit_line(xs, ys)
regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = (m * predict_x) + b

ys_data = ys
ys_regline = [(m*x) + b for x in xs]

r_squared = coefficient_of_determination(ys_data, ys_regline)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs, regression_line, color = 'b')
plt.show()
  
    
    
    
    
    
    