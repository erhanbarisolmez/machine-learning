import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('btk-makine/data/ads_ctr_optimisation.csv')

import random
N=10000 # 10.000 tıklama
d=10 #toplam 10 ilan
#Ri(n)
oduller=[0] *d # ilk baştaki bütün ilanların ödülü 0
#Ni(n)
tiklamalar = [0] *d #o ana kadarki tıklamalar 
toplam=0 #toplam ödül
secilenler=[]

for n in range(0, N):
    ad = 0  # seçilen ilan
    max_ucb = 0
    for i in range(0, d):
        if (tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3 / 2 * math.log(n) / tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:  # max'tan büyük bir ucb çıkınca
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n, ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print("toplam ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()