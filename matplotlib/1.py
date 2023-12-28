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
plot(x,y): x ve y değerleri arasındaki ilişkiyi çizmek için kullanılır.
scatter(x,y): Nokta bulutu oluşturmak için kullanılır.
bar(x, height): Sütun grafik oluşturmak için kullanılır.
stem(x,y): Gövde grafiği oluşturmak için kullanılır.
fill_between(x,y1,y2): : İki eğri arasını dolduran alanı göstermek için kullanılır.
stackplot(x,y): Yığılmış alan grafiği oluşturmak için kullanılır.
stairs(values): Basamaklı çizgi grafiği oluşturmak için kullanılır.
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


""" Pairwise Data - fill_between(x,y1,y2) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(1)
x= np.linspace(0,8,16)
y1 = 3+4*x/8 + np.random.uniform(0.0, 0.5, len(x))
y2 = 1 + 2 * x/8 + np.random.uniform(0.0, 0.5, len(x))

#plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/ 2, linewidth=2)

ax.set(xlim = (0, 8), xticks = np.arange(1,8),
       ylim = (0, 8), yticks=np.arange(1,8))

plt.show()

""" Pairwise Data - stackplot(x,y) """
import matplotlib.pyplot as plt
import numpy as np

#make data
x = np.arange(0,10,2)
ay = [1, 1.25, 2, 2.75, 3]
by = [1,1,1,1,1]
cy = [2,1,2,1,2]
y = np.vstack([ay, by, cy])

#plot
fig, ax = plt.subplots()

ax.stackplot(x,y)

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks= np.arange(1,8))

plt.show()

""" Pairwise Data - stairs(values) """
import matplotlib.pyplot as plt
import numpy as np

#make data
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

#plot
fig, ax = plt.subplots()

ax.stairs(y, linewidth = 2.5)

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Statistical Distributions (İstatistiksel Veriler)
-----------------------------------------------------
hist(x): Histogram oluşturmak için kullanılır.
boxplot(X): Kutu grafiği oluşturmak için kullanılır.
errorbar(x,y,yerr,xerr): Hata çubukları ile grafik oluşturmak için kullanılır.
violinplot(D): Keman grafiği oluşturmak için kullanılır.
eventplot(D): Olay grafiği oluşturmak için kullanılır.
hist2d(x,y): İki değişkenin 2D histogramını oluşturmak için kullanılır.
hexbin(x,y,C):  İki değişkenin hexbin grafiğini oluşturmak için kullanılır.
pie(x): Pasta grafiği oluşturmak için kullanılır.
ecdf(x): İmpirikik Kümülatif Dağılım Fonksiyonunu oluşturmak için kullanılır.
"""


""" Statistical Distributions - hist(x) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

#plot
fig, ax = plt.subplots()

ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

ax.set(xlim = (0,8), xticks = np.arange(1,8), 
       ylim = (0,56), yticks = np.linspace(0,56,9))

plt.show()

""" Statistical Distributions - boxplot(X) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(10)
D = np.random.normal((3,5,4), (1.25, 1.00, 1.25), (100,3))

#plot
fix, ax = plt.subplots()
VP = ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})

plt.show()

""" Statistical Distributions - errorbar(x, y, yerr, xerr) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(1)
x = [2,4,6]
y = [3.6, 5, 4.2]
yerr=[0.9, 1.2,0.5]

#plot
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Statistical Distributions - violinplot(D) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(10)
D = np.random.normal((3,5,6), (0.75, 1.00, 0.75), (200, 3))
# D: Violin plot çizilecek veri
# [2, 4, 6]: Violin plot'ların konumları
# widths=2: Violin plot'ların genişliği
# showmeans, showmedians, showextrema: Ortalama, medyan, ve uç noktaların gösterilip gösterilmeyeceği

#plot
# Şekil oluşturma
fig, ax = plt.subplots()

vp = ax.violinplot(D, [2,4,6], widths=2,
                   showmeans=False, showmedians=False, showextrema=False)
# D: Violin plot çizilecek veri
# [2, 4, 6]: Violin plot'ların konumları
# widths=2: Violin plot'ların genişliği
# showmeans, showmedians, showextrema: Ortalama, medyan, ve uç noktaların gösterilip gösterilmeyeceği

#styling
for body in vp['bodies']:
  body.set_alpha(0.9) # Violin plotların şeffaflığı

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))
# x ekseni sınırları ve işaretlenmiş konumlar
# y ekseni sınırları ve işaretlenmiş konumlar

plt.show()

""" Statistical Distributions - eventplot(D) """
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

#make data
np.random.seed(1)
x = [2,4,6]
D = np.random.gamma(4, size=(3,50))

#plot
fig, ax = plt.subplots()

ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidths=0.75)

ax.set(xlim=(0,8), xticks=np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Statistical Distributions - hist2d(x,y) """
#make data: correlated + noise 
np.random.seed(1)
x = np.random.randn(5000)
y = 1.2 * x + np.random.randn(5000) / 3

#plot 
fig, ax = plt.subplots()

ax.hist2d(x, y, bins=(np.arange(-3,3,0.1), np.arange(-3, 3, 0.1)))

ax.set(xlim=(-2,2), ylim=(-3, 3))

plt.show()

""" Statistical Distributions - hexbin(x,y,C) """
import matplotlib.pyplot as plt
import numpy as np

#make data
np.random.seed(1)
x = np.random.randn(5000)
y = 1.2 * x + np.random.randn(5000) / 3

#plot
fig, ax = plt.subplots()

ax.hexbin(x,y, gridsize=20)
ax.set(xlim=(-2, 2), ylim=(-3, 3))

plt.show()

""" Statistical Distributions - pie(x) """
import matplotlib.pyplot as plt
import numpy as np

#make data
x = [1, 2, 3 ,4]
colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))

#plot
fig, ax = plt.subplots()
ax.pie(x, colors=colors, radius=3, center=(4,4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

ax.set(xlim=(0,8), xticks = np.arange(1,8),
       ylim=(0,8), yticks=np.arange(1,8))

plt.show()

""" Statistical Distributions - ecdf(x) """
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

ecdf = sm.distributions.ECDF(x)

#plot
fig, ax = plt.subplots()
ax.step(ecdf.x, ecdf.y, label='ECDF')

ax.set(xlabel="x", ylabel= "Cumulative Probability", title="Empirical Cumulative ")
ax.legend(loc='best')
plt.show()
