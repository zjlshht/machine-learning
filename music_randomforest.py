from pandas.io.parsers import read_csv
from funtools import divided_data, divided_y, datamini
A = read_csv("Data_on_year.csv", index_col=0).values
X = A[:, :13]
y = A[:, 13]
y = divided_y(y)
X_train, X_test, y_train, y_test = divided_data(X, y, standard=False)
y_pred = datamini.ranforest(X_train, X_test, y_train, threshold=False)
