#https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
#import csv
'''可以运行  但是和书上结果不对'''
import numpy as np
from sklearn import cluster, covariance, manifold
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
'''
#let's load the input data from the wholesale.csv file that's already provided to you:
    # load data from input file
input_file='Wholesale.csv'
file_reader=csv.reader(open(input_file,'rb'),delimiter=',') #这个文件用read_csv读取
#是dataframe，这个读取是reader文件
X=[]
for count,row in enumerate(file_reader):
    if not count:
        names=row[2:]
        continue
    X.append([float(x) for x in row[2:]])
X=np.array(X)
'''
from pandas.io.parsers import read_csv
file=read_csv("wholesale.csv")
X=file.values
#let's build a mean shift model like we did in one of the ealier recipes:
    #estimating the bandwidth
bandwidth=estimate_bandwidth(X,quantile=0.8,n_samples=len(X))

#compute clustering with Meanshift
meanshift_estimator=MeanShift(bandwidth=bandwidth,bin_seeding=True)
meanshift_estimator.fit(X)
labels=meanshift_estimator.labels_
centroids=meanshift_estimator.cluster_centers_
num_clusters=len(np.unique(labels))

print("\nNumber of clusters in input data =",num_clusters)

#let's print the centroids of clusters that we obtained,as follows:
#print("\nCentroids of clusters:")
#print('\t'.join([name[:3] for name in names]))
for centroid in centroids:
    print('\t'.join([str(int(x)) for x in centroid]))

#let's visualize a couple of features to get a sense of the output:
    #visualizing data

centroids_milk_groceries=centroids[:,1:3]

#plot the nodes using the coordinates of our centroids_milk_groceries
plt.figure()
plt.scatter(centroids_milk_groceries[:,0],centroids_milk_groceries[:,1],
            s=100,edgecolors='k',facecolors='none')

offset=0.2
plt.xlim(centroids_milk_groceries[:,0].min()-offset*centroids_milk_groceries[:,0].ptp(),
         centroids_milk_groceries[:,0].max()+offset*centroids_milk_groceries[:,0].ptp(),)

plt.ylim(centroids_milk_groceries[:,1].min()-offset*centroids_milk_groceries[:,1].ptp(),
         centroids_milk_groceries[:,1].max()+offset*centroids_milk_groceries[:,1].ptp())

plt.title('Centroids of clusters for milk and groceries')
plt.show()