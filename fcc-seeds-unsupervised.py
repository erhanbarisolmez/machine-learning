import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

cols =["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("data/seeds_dataset.txt", names = cols, sep="\s+")
df.head()

for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label = cols[j]
        sns.scatterplot(x=x_label, y=y_label, data=df, hue='class')
        plt.show()

# Clustering

from sklearn.cluster import KMeans

x = "perimeter"
y = "asymmetry"
X = df[[x,y]].values

kmeans = KMeans(n_clusters=3, n_init=10).fit(X)

clusters = kmeans.labels_

cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x,y, "class"])

# K Means classes
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()

# Original classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()

# Higher Dimensions
X = df[cols[:1]].values

kmeans = KMeans(n_clusters=3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1,1))), columns=df.columns)

sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()

sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()
 