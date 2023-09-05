import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('data/odev_tenis.csv')
print(veriler)

#encoder
# play = veriler.iloc[:,-1].values
# print(play)

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()
# play = le.fit_transform(veriler.iloc[:,-1])
# print(play)

# windy = le.fit_transform(veriler.iloc[:,-1])
# print(windy)

#kısa yol
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c= veriler2.iloc[:,:1] # :, ilk satırın hepsini al , :1 -> 1 e kadar olanını al


from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data=c, index=range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)
sonveriler = pd.concat([sonveriler, veriler2.iloc[:-2:], sonveriler],axis=1)


#verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


import statsmodels.api as sm 

