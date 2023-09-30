import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/veriler.csv")
print(data)

x = data.iloc[:,1:4].values #bağımsız değişken
y = data.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)

#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train) #  fit:eğitme transform: uygulama
X_test = sc.transform(x_test)

# obje oluşturuyoruz
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)