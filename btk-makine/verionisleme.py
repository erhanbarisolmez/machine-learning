import pandas as pd # verileri tutma ve erişme
import numpy as np  # büyük sayılar ve hesaplama işlemleri 
import matplotlib.pyplot as plt # çizimler 
from sklearn.impute import SimpleImputer  #obje tanımlama boş alanları değiştircez

# veri yükleme 
data=pd.read_csv('data/satislar.csv')
print(data)

#veri ön işleme
aylar = data[['Aylar']]
print(aylar)

satislar = data[['Satislar']]
print(satislar)

satislar2 = data.iloc[:,:1].values
print(satislar2)

#data bölme  ve test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33, random_state=0)

# farklı dünyada olan verileri aynı dünyada kullanma öznitelik ölçekleme

# from sklearn.preprocessing import StandardScaler
 
# sc= StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

# Y_train = sc.fit_transform(y_train)
# y_test = sc.fit_transform(y_test)

# model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title('aylara göre satış')
plt.xlabel('aylar')
plt.ylabel('satislar')   

# cinsiyet için  dönüşüm
c = data.iloc[:,-1].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(data.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)