import numpy as np
import pandas as pd
yorumlar = pd.read_csv('btk-makine/data/Restaurant_Reviews.csv')

import re
import nltk

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

derlem=[]
for i in range(1000):
  yorum = re.sub('[^a-Za-Z]',' ',yorumlar['Review'][i]) 
  yorum = yorum.lower()
  yorum = yorum.split()
  yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
  yorum = ' '.join(yorum)
  derlem.append(yorum)