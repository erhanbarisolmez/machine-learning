import pandas as pd # verileri tutma ve erişme
import numpy as np  # büyük sayılar ve hesaplama işlemleri 
import matplotlib.pyplot as plt # çizimler 
from sklearn.impute import SimpleImputer  #obje tanımlama boş alanları değiştircez

# veri yükleme 
data=pd.read_csv('data/satislar.csv')
print(data)

# farklı dünyada olan verileri aynı dünyada kullanma öznitelik ölçekleme

# from sklearn.preprocessing import StandardScaler
 
# sc= StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

# Y_train = sc.fit_transform(y_train)
# y_test = sc.fit_transform(y_test)



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