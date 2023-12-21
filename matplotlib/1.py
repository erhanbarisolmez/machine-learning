import matplotlib.pyplot as plt
import numpy as np

# Örnek 1:
x = np.linspace(0, 2 * np.pi,200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()

"""
#Matplotlib ile oluşturabileceğiniz grafik türleri#
  
  İkili Veri: ikili grafikleri(x,y), tablo halinde(var_0, ..., var_n) ve işlevsel f(x) = yveri
  
  İstatistiksel Dağılımlar: Bir veri kümesindeki en az bir değişkenin dağılım grafikleri. Bu yöntemlerden bazıları dağılımları da hesaplar
  
  Izgaralı Veriler: Dizilerin ve görüntülerin grafikleri(Zi,j)  ve alanlar (Ui,j,Vi,j)
düzenli ızgaralarda ve karşılık gelen koordinat ızgaralarında (Xi, j, Yi,j)
  
  Düzensiz Gridlenmiş Veri: Veri grafikleri (Zx,y)
yapılandırılmamış ızgaralarda , yapılandırılmamış koordinat ızgaralarında(x,y)
ve 2D işlevler f(x,y) = z

  3D ve Hacimsel Veriler: Üç boyutlu grafikler (x,y,z), yüzey ( f(xy)=z )ve hacimsel Vx,y,z
Kütüphaneyi kullanarak veriler mpl_toolkits.mplot3d.

"""

""" 
Pairwise Data (İkili Veriler)
-------------------------------
plot(x,y):
scatter(x,y):
bar(x, height)
stem(x,y)
fill_between(x,y1,y2):
stackplot(x,y):
stairs(values):
"""


""" Pairwise Data -  plot(x,y) """
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

#make data
x = np.linspace(0,10,100)
y = 4 + 2 - np.sin(2*x)

# plot
fig, ax = plt.subplots()

ax.plot(x,y, linewidth=2.0)

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Pairwise Data - scatter(x,y) """
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

#make data 
np.random.seed(3)
x = 4 + np.random.normal(0,2,24)
y = 4 + np.random.normal(0,2,len(x))

#size and color
sizes = np.random.uniform(15,80, len(x))
colors = np.random.uniform(15,80, len(x))

#plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0,8), xticks = np.arange(1,8),
       ylim=(0,8), yticks = np.arange(1,8))

plt.show()

""" Pairwise Data - bar(x,height) """
import matplotlib.pyplot as plt
import numpy as np

#make data
x = 0.5 + np.arange(8)
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

#plot
fig, ax = plt.subplots()

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0,8), xticks = np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Pairwise Data - stem(x,y) """
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery') 

#make data
x = 0.5 + np.arange(8)
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

#plot
fig, ax = plt.subplots()

ax.stem(x,y)

ax.set(xlim = (0,8), xticks = np.arange(1,8),
       ylim = (0,8), yticks = np.arange(1,8))

plt.show()





