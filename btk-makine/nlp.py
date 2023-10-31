import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
yorumlar = pd.read_csv('btk-makine/data/Restaurant_Reviews.csv')

import re
import nltk

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer

ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (önişleme)
derlem=[]
for i in range(1000):
  yorum = re.sub('[^a-zA-Z]',' ',yorumlar['ReviewLiked'][i]) 
  yorum = yorum.lower()
  yorum = yorum.split()
  yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
  yorum = ' '.join(yorum)
  derlem.append(yorum)

#Feautre Exraction ( Öznitelik çıkarımı)
#Bag of Words (BOW)
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features=1000)
  X = cv.fit_transform(derlem).toarray() #bağımsız değişken
  y = yorumlar.iloc[0:,0].values #bağımsız değişken


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)