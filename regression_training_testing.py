import pandas as pd
import math
import quandl
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import sklearn.model_selection


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Open'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT','PCT_Change','Adj. Volume']] #features

forecast_col = 'Adj. Close'
df.fillna('-99999',inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))  #1% shift of data

df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# print(df.head())

X = np.array(df.drop(['Label'],1)) #Features
Y = np.array(df['Label'])
X = preprocessing.scale(X)
Y = np.array(df['Label'])



X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X,Y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)


print(accuracy)

#Linear regression is squared error
#n_jobs gives multithreading capabilities
#SVM has kernals
