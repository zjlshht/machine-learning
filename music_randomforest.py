"""
This is a example to run model randomforest without threshold to classify
"""
from pandas.io.parsers import read_csv
from utils import divided_data, divided_y, datamini, data_visualize
A = read_csv("Data_on_year.csv", index_col=0).values
X = A[:, :13]
y = A[:, 13]
y = divided_y(y)
X_train, X_test, y_train, y_test = divided_data(X, y, standard=False)
model = datamini(X_train, X_test, y_train, use_threshold=False)
y_pred = model.ranforest()
threshold_portion = y_train.mean()
plot = data_visualize(y_test, y_pred, threshold_portion, 'RandomForest')
plot.ROC_curve()
plot.PR_curve()
plot.COST_curve()
plot.indicators_table()
