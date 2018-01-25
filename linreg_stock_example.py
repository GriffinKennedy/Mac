# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot') #have your matplotlib look like ggplot

df = quandl.get('WIKI/GOOGL') #use quandl for stock data this is how you read it in

#print(df.head())

### we only want the code that adds value some of the columns kind of give repeat info
### so let's choose and create our columns
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df['HL_PCT'] = (df['Adj. High'] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

# This is what we dubbed of value
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df.head())

forcast_col = "Adj. Close"
df.fillna(-99999, inplace=True) #we are replacing our na's with outlier data this is a good option for ML
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forcast_col].shift(-forecast_out)


x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]


df.dropna(inplace = True)

y = np.array(df['label'])
y = np.array(df['label'])


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size = 0.2)
clf = LinearRegression(n_jobs = -1) #can also try svm.SVR()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
#print(accuracy) #is the squared error
#print(forecast_out) #how many days in advance

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()