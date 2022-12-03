#-------------------------------------------------------------------------
# AUTHOR: Miro Abdalian
# FILENAME: Clustering HW5-2
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('./HW5-2/training_data.csv', sep=',', header=None) #reading the data by using Pandas library
#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:]
#run kmeans testing different k values from 2 until 20 clusters

max_coeff = 0
best_k = 2
k_data = []
coeff_data = []

for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)
     
     #for each k, calculate the silhouette_coefficient
     silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
     k_data.append(k)
     coeff_data.append(silhouette_coefficient)
     #find which k maximizes the silhouette_coefficient
     if silhouette_coefficient > max_coeff:
          max_coeff = silhouette_coefficient
          best_k = k

print(f"k = {best_k} maximizes the silhouette_coefficient" )

x = np.array(k_data)
y = np.array(coeff_data)
 
# plot the value of the silhouette_coefficient for each k 
plt.title("Line graph")
plt.xlabel("k value")
plt.ylabel("silhouette coeff")
plt.plot(x, y, color ="green")
plt.xticks(x)
plt.show()

#reading the test data (clusters) by using Pandas library
df = pd.read_csv('./HW5-2/testing_data.csv', sep=',', header=None) #reading the data by using Pandas library
labels = np.array(df.values).reshape(1, 3823)[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
