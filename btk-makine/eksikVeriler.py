import pandas as pd # verileri tutma ve erişme
import numpy as np  # büyük sayılar ve hesaplama işlemleri 
import matplotlib.pyplot as plt # çizimler 
from sklearn.impute import SimpleImputer  #obje tanımlama boş alanları değiştircez

# veri yükleme 
data=pd.read_csv('data/eksik-veriler.csv')

print(data)

#veri ön işleme
boy = data[['boy']]
print(boy)


boyKilo = data[['boy','kilo']]
print(boyKilo)

class insan:
    boy=190
    def kosmak(self,b):
        return b+10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l = [1,3,4] 

#eksik veriler 

imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = data.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

# kategorik veriler - kategorik veriyi sayısal veriye dönüştürme işlemi

ulke = data.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(data.iloc[:,-1])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

c = data.iloc[:, -1 ].values
print(c)

from sklearn import preprocessing

le = preprocessing.label.fit_transform(data.iloc[:,0])
print(c)

#data birleştirme
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr','us'])
print(sonuc)
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)
cinsiyet = data.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)
s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

# data bölme 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, test_size=0.33, random_state=0)

# farklı dünyada olan verileri aynı dünyada kullanma öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
