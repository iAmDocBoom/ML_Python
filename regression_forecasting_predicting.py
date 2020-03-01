import pandas as pd
import math, datetime
import quandl
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Open'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT','PCT_Change','Adj. Volume']] #features

forecast_col = 'Adj. Close'
df.fillna('-99999',inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))  #1% shift of data

df['Label'] = df[forecast_col].shift(-forecast_out) #forecasting 30 days


# print(df.head())

X = np.array(df.drop(['Label'],1)) #Features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]



df.dropna(inplace=True)
Y = np.array(df['Label'])
Y = np.array(df['Label'])



X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X,Y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)

forecast_set = clf.predict(X_lately)
# print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #number of seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# print(accuracy)

#Linear regression is squared error
#n_jobs gives multithreading capabilities
#SVM has kernals