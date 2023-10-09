import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

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

# Sınıflandırma Algoritmaları...

#LogisticRegression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train) #egitim

y_pred = logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)
# verimiz ile tahmin verisini kontrol ediyor ve hata oranını söylüyor(karmaşıklık matrisi)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#KNN algoritması
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train) #eğitim

y_pred = knn.predict(X_test) #tahmin
#karmaşıklık matrisi
cm= confusion_matrix(y_test, y_pred)
print(cm)

# SVC
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train) # eğitim 

y_pred = svc.predict(X_test) # tahmin
# karmaşıklık matrisi ile test veri ile kıyaslama 
cm = confusion_matrix(y_test, y_pred)
print("SVC")
print(cm)

#naif bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) #eğitim

y_pred = gnb.predict(X_test) #tahmin 
cm = confusion_matrix(y_test, y_pred) #sonuç test
print('GNB')
print(cm)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train) # eğitim
y_pred = dtc.predict(X_test) # tahmin

cm = confusion_matrix(y_test, y_pred) #test
print('DTC')
print(cm)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train) #eğitim
y_pred = rfc.predict(X_test) #tahmin
cm = confusion_matrix(y_test, y_pred) #test
print('RFC')
print(cm)
print(y_proba[:,0])

# ROC, TPR, FPR
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])
from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='e')
print(fpr)
print(tpr)