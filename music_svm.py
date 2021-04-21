from pandas.io.parsers import read_csv
from funtools import divided_data, divided_y, datamini
import numpy as np
A = read_csv("Data_on_year.csv", index_col=0).values
random_list = np.random.choice(len(A), 5000)
A_sample = A[random_list]
X = A_sample[:, :13]
y = A_sample[:, 13]
y = divided_y(y)
X_train, X_test, y_train, y_test = divided_data(X, y, standard_num=13)
y_pred = datamini.svm(X_train, X_test, y_train)
