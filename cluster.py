#https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
import csv
import numpy as np
from sklearn import cluster, covariance, manifold
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

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
#let's build a mean shift model like we did in one of the ealier recipes:
    #estimating the bandwidth
