"""
1. Doğrusal Regresyon:
Doğrusal regresyon, bağımlı değişkenle bağımsız değişken arasındaki doğrusal ilişkiyi modellemek için kullanılır. 
Örnek olarak, bir evin metrekare cinsinden alanı (X) ile fiyatı (Y) arasındaki ilişkiyi modellemek istediğimizi düşünelim.
"""

# 1. Doğrusal Regresyon 

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"Veri Oluşturma"
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100,1)

"Modeli Eğitme"
model =  LinearRegression()
model.fit(X, Y)

"Eğitilmiş Model Kullanılarak Tahmin Yapma"
X_new = np.array([[0], [2]])
Y_pred = model.predict(X_new)

"Sonuçları Görselleştirme"
plt.scatter(X,Y, color="blue")
plt.plot(X_new, Y_pred, color="red", linewidth=3)
plt.xlabel('Metrekare Alanı')
plt.ylabel('Fiyat')
plt.title('Doğrusal Regresyon Örneği')
plt.show()

"""
Polinom regresyon, doğrusal olmayan ilişkileri modellemek için kullanılır. 
Örneğin, bir bağlantı noktasındaki bir aracın hızının (X) zamana bağlı olarak değiştiğini düşünelim
"""

# 2 .Polinom Regresyon 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

"Veri Oluşturma"
np.random.seed(0)
X = 6* np.random.rand(100, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

"Degree'yi belirtip PolinomFeature ekleme - Veriyi dönüştürme (polinom özellik ekleme)"
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

"Modeli Eğitme"
model = LinearRegression()
model.fit(X_poly, Y)

"Eğitilmiş modeli kullanarak tahmin yapma"
X_new = np.linspace(-3, 3, 100).reshape(100,1)
X_new_poly = poly_features.transform(X_new)
Y_pred = model.predict(X_new_poly)

"Grafiksel gösterim"
plt.scatter(X,Y, color="blue")
plt.plot(X_new, Y_pred, color="red",linewidth=3)
plt.xlabel('Hız')
plt.ylabel('Bağlantı Noktasındaki Araç Sayısı')
plt.title('Polinom Regresyon Örneği')
plt.show()

""""
Gaussian regresyon, bir noktanın konumuna göre diğer noktaların olasılık yoğunluğunu tahmin etmek için kullanılır. 
Örneğin, bir konumdaki bir restoranın müşteri sayısını modellemek istediğimizi düşünelim.
"""
# 3. Gaussian Regresyon:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

"Veri Oluşturma"
np.random.seed(0)
X = np.sort(5* np.random.rand(100, 1), axis=0)
Y = np.sin(X).ravel()

"Gaussian Regresyon Modelini Tanımlama"
kernel = C(1.0, (1e-3, 1e3) * RBF(1.0, (1e-2, 1e2)))

"MODELİ EĞİTME"
model.fit(X,Y)

"Eğitilmiş modeli kullanarak tahmin yapma"
X_new = np.linspace(0, 5, 100).reshape(-1, 1)
Y_pred, sigma = model.predict(X_new, return_std=True)

"Sonuçları görselleştirme"
plt.scatter(X, Y, color='blue')
plt.plot(X_new, Y_pred, color='red', linewidth=3)
plt.fill_between(X_new.ravel(), Y_pred - sigma, Y_pred + sigma, color='pink', alpha=0.3)
plt.xlabel('Konum')
plt.ylabel('Müşteri Sayısı')
plt.title('Gaussian Regresyon Örneği')
plt.show()

"""
Üssel regresyon, bağımsız değişkenin üslerini içeren bir denklem kullanarak bir model oluşturur. 
Örneğin, bir popülasyonun zaman içindeki büyümesini modellemek için üssel regresyon kullanabiliriz.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Veri oluşturma
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 3 + 4 * X + 1.5 * X**2 + np.random.randn(100, 1)

# Veriyi dönüştürme (üsler eklemek)
X_exp = np.c_[X, X**2]

# Modeli eğitme
model = LinearRegression()
model.fit(X_exp, Y)

# Eğitilmiş modeli kullanarak tahmin yapma
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_exp = np.c_[X_new, X_new**2]
Y_pred = model.predict(X_new_exp)

# Sonuçları görselleştirme
plt.scatter(X, Y, color='blue')
plt.plot(X_new, Y_pred, color='red', linewidth=3)
plt.xlabel('Bağımsız Değişken')
plt.ylabel('Bağımlı Değişken')
plt.title('Üssel Regresyon Örneği')
plt.show()
