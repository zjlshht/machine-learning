from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from prettytable import PrettyTable
import numpy as np


class datamini():
    """
    this class contain the model with given Hyperparameter
    -------
    threshold refer to whether use additional value instead of portion
    use_threshol determines whether y_pred should be classified according to the threshold
    """

    def __init__(self, X_train, X_test, y_train, threshold=False, use_threshold=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.threshold = threshold
        self.use_threshold = use_threshold

    def xgboost(self):
        if not self.threshold:
            threshold = self.y_train.mean()
        else:
            threshold = self.threshold
        model = XGBClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        if not self.use_threshold:
            return y_pred
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        return y_pred

    def FNNetwork(self):
        if not self.threshold:
            threshold = self.y_train.mean()
        else:
            threshold = self.threshold
        model = Sequential()
        model.add(Dense(units=20, activation='sigmoid', input_dim=13))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=10, batch_size=32)
        prediction = model.predict(self.X_test)
        if not self.use_threshold:
            return prediction
        prediction[prediction >= threshold] = 1
        prediction[prediction < threshold] = 0
        return prediction

    def KNN(self, k):
        if not self.threshold:
            threshold = self.y_train.mean()
        else:
            threshold = self.threshold
        clf = KNeighborsRegressor(algorithm='kd_tree', n_neighbors=k)
        clf.fit(self.X_train, self.y_train)
        prediction = clf.predict(self.X_test)
        if not self.use_threshold:
            return prediction
        prediction[prediction > threshold] = 1
        prediction[prediction <= threshold] = 0
        return prediction

    def dectree(self):
        clf = DecisionTreeClassifier(
            random_state=666, min_impurity_decrease=1e-2)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return y_pred

    def ranforest(self):
        if not self.threshold:
            threshold = self.y_train.mean()
        else:
            threshold = self.threshold
        clf = RandomForestClassifier(
            random_state=666, min_impurity_decrease=1e-4)
        clf.fit(self.X_train, self.y_train)
        y_pro = clf.predict_proba(self.X_test)
        y_pred = np.zeros(len(y_pro))
        if not self.use_threshold:
            for i in range(len(y_pro)):
                y_pred[i] = y_pro[i][1]
            return y_pred
        for i in range(len(y_pro)):
            if y_pro[i][1] > threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

    def svm(self):
        if not self.threshold:
            threshold = self.y_train.mean()
        else:
            threshold = self.threshold
        clf = SVR(kernel='poly', degree=2, C=1.0, epsilon=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        if not self.use_threshold:
            return y_pred
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        return y_pred


class data_visualize():
    """
    This class reads parameters and make data visualization
    -------
    name refer to model name
    threshold_portion refer to the threshold that given by portion
    beta refer to weight in recall and precision
    fpr_cost_index prefer to make cost curve more lively when y_pred have many different values
    divided_by_threshold refer to whether the y_pred has been classified according to the threshold
    """

    def __init__(self, y_test, y_pred, threshold_portion, name, beta=3, fpr_cost_index=200, divided_by_threshold=False):
        self.y_test = y_test
        self.y_pred = y_pred
        self.threshold_portion = threshold_portion
        self.name = name
        self.beta = beta
        self.table_cost = divided_by_threshold
        self.precision, self.recall, self.thresholds = precision_recall_curve(
            self.y_test, self.y_pred)
        if not divided_by_threshold:
            self.fpr, self.tpr, self.threshold = roc_curve(
                self.y_test, self.y_pred)
            self.pr_index = find_index(self.thresholds, self.threshold_portion)
            self.Precision = self.precision[self.pr_index]
            self.Recall = self.recall[self.pr_index]
            self.roc_auc = auc(self.fpr, self.tpr)
            self.cost_y, self.cost_auc = cost_curve(
                self.fpr, self.tpr, fpr_cost_index)
        else:
            self.Recall = self.recall[1]
            self.Precision = self.precision[1]
        self.F1_score = 2*self.Precision * \
            self.Recall/(self.Precision+self.Recall)
        self.F_beta_score = (1+self.beta**2) / \
            (1/self.Precision+self.beta**2/self.Recall)

    def ROC_curve(self):
        """
        plot roc curve
        """
        plt.plot(self.fpr, self.tpr)
        plt.title("ROC curve of "+"{}".format(self.name))
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.show()

    def PR_curve(self):
        """
        plot pr curve
        """
        plt.plot(self.recall, self.precision)
        plt.title("PR curve of "+"{}".format(self.name))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.show()

    def COST_curve(self):
        """
        plot cost curve
        """
        plt.plot(np.linspace(0, 1, 100), self.cost_y)
        plt.title("COST curve of "+"{}".format(self.name))
        plt.xlabel("Positive probability cost")
        plt.ylabel("Normalized cost")
        plt.show()

    def indicators_table(self):
        """
        Return a table that contain various indicators
        -------
        if prediction is given by threshold, figure can not
        be drawn, therefore there is no auc in the table

        """
        table = PrettyTable(["indicators", "values"])
        table.add_row(["Precision", self.Precision])
        table.add_row(["Recall", self.Recall])
        table.add_row(["F1_score", self.F1_score])
        table.add_row(["F_beta_score", self.F_beta_score])
        if not self.table_cost:
            table.add_row(["roc_auc", self.roc_auc])
            table.add_row(["cost_auc", self.cost_auc])
        print(table)


def filter_by_threshold(y_pred, threshold):
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0
    return y_pred


def divided_y(y, threshold=0.6):
    for i in range(len(y)):
        if y[i] >= threshold:  # 0.6为分界线
            y[i] = 1
        else:
            y[i] = 0
    return y


def divided_data(X, y, seed=666, test_size=0.25, standard=True, standard_num=11):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    if standard:
        scaler = StandardScaler()
        scaler.fit(X_train[:, :standard_num])
        X_train[:, :standard_num] = scaler.transform(X_train[:, :standard_num])
        X_test[:, :standard_num] = scaler.transform(X_test[:, :standard_num])
    return [X_train, X_test, y_train, y_test]


def find_index(array, member):
    if array[0] > member:
        for i in range(len(array)):
            if array[i] <= member:
                return i
    else:
        for i in range(len(array)):
            if array[i] >= member:
                return i


def cost_curve(fpr, tpr, k=1):
    y_point = []
    auc = 0
    for j in np.linspace(0, 1, 100):
        min_number = min(map(lambda x, y: x+j*(y-x), fpr[k:-k], tpr[k:-k]))
        y_point.append(min_number)
    for i in range(99):
        auc += (y_point[i]+y_point[i+1])/2
    auc /= 100
    return [y_point, auc]
