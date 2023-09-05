import pandas as pd # verileri tutma ve erişme
import numpy as np  # büyük sayılar ve hesaplama işlemleri 
import matplotlib.pyplot as plt # çizimler 

data=pd.read_csv('/data/veriler.csv')

print(data)

boy = data[['boy']]
print(boy)

boyKilo = data[['boy','kilo']]
print(boyKilo)

class insan:
    boy=190
    def kosmak(self,b):
        return b+10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l = [1,3,4] 

