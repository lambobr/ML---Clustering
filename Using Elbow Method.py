import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(10,4))
#generate samples
n_samples=1500
random_state=170
X,y=make_blobs(n_samples=n_samples,random_state=random_state)

#find best number of clusters using elbow method
inertia=[]
for i in range(1,11):
    z=KMeans(n_clusters=i,random_state=random_state).fit(X)
    inertia.append(z.inertia_)
plt.subplot(121)
plt.plot(range(1,11),inertia)
plt.title('K-Means Clustering (Elbow Method)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia / Distortion / Cost Function')
#From the plot we see that the optimum number of clusters is 3


#Fitting into the model
kmeans=KMeans(n_clusters=3,random_state=random_state)
y_pred=kmeans.fit_predict(X)
plt.subplot(122)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],c='black')
plt.title('K-Means Clustering Model')
plt.grid()