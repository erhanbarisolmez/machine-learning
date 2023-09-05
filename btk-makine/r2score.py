
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
# veri yükleme
veriler = pd.read_csv('data/maaslar.csv')

# data kolonlarını aldık
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

# linear regression x ve ye değerlerini böldük
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()#obje oluşturduk
lin_reg.fit(X,Y)

plt.scatter(X,Y, color= 'red')
plt.plot(x,lin_reg.predict(X), color= 'blue')
plt.show()

print('linear regression R2 değeri')
print(r2_score(Y,lin_reg.predict(X)))

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()#obje oluşturduk
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.show()

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()#obje oluşturduk
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.show()

#tahminler 
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print('polynomial regression R2 değeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

#verilerin ölçeklendirilmesi 
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y) 

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue')

plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color='blue')

print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

print('SVR R2 değeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))



#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color = "red")
plt.plot(x, r_dt.predict(X), color = "blue") #her bir X değeri için tahmin değerini kullan

plt.plot(x,r_dt.predict(Z), color = "green")
plt.plot(x,r_dt.predict(K), color = "yellow")

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print('Decision Tree R2')
print(r2_score(Y,r_dt.predict(X)))

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0) #10 çizim 
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color='red')
plt.plot(X,rf_reg.predict(X), color = 'blue')

plt.plot(X,rf_reg.predict(Z), color ='green')

#r2score

print('Random Forest R2 değeri')
print(r2_score(Y,rf_reg.predict(X)))

print('Random Forest R2 değeri')
print(r2_score(Y,rf_reg.predict(K)))

print('Random Forest R2 değeri')
print(r2_score(Y,rf_reg.predict(Z)))

#Özer r2 değerleri