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
birler = [0] * d
sifirlar = [0] *d 

for n in range(1, N):
    ad = 0  # seçilen ilan
    max_th = 0
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] +1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad] + 1
    odul = veriler.values[n, ad]
    if odul == 1:
        birler[ad] = birler[ad]+1
    else : 
        sifirlar[ad] = sifirlar[ad] +1
    toplam = toplam + odul

print("toplam ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()