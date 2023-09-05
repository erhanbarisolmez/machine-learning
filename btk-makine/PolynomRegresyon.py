import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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