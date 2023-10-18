import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('btk-makine/data/ads_ctr_optimisation.csv')
print(veriler)

import random 

N= 1000
d = 10
toplam = 0
secilenler=[]
for n in range (0,N):
  ad = random.randrange(d)
  secilenler.append(ad)
  odul = veriler.values[n,ad] #verilerdeki n. satır = 1 ise ödül +1 
  toplam = toplam + odul

plt.hist(secilenler)
plt.show()
