"""
  Grid Search:İdeal hiperparametre setini tanımlamamızı sağlayan, makine öğrenimi modellerimizin hassasiyetini artıran ve hatasını azaltan bir teknik.
  Uygulamanın ardından kullanabileceğimiz en iyi parametreleri ve alabileceğimiz en iyi puanı modelimizde görebiliriz.
"""

# Importing all the necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Reading our dataset
df = pd.read_csv('heart.csv')
# Independent variables
X = df.drop('target', axis = 1)
# Target variable 
y = df['target']

# Splitting our dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Random Forest Classifier
rfc = RandomForestClassifier()

# Limit of parameters
forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]

# GridSearchCV
clf = GridSearchCV(rfc, forest_params, cv = 10, scoring='accuracy')

clf.fit(X_train, y_train)

# Printing our best metrics
print(clf.best_params_)

print(clf.best_score_)
