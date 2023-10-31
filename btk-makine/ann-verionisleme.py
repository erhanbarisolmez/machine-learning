import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

veriler = pd.read_csv('btk-makine/data/Churn_Modelling.csv')
print(veriler)

#veri ön işleme
X = veriler.iloc[:,3:13].values 
Y = veriler.iloc[:,13].values

#encoder : Kategorik -> Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])], 
      remainder = "passthrough"
    )
X = ohe.fit_transform(X)
X = X[:, 1:]