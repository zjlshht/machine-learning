from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


class datamini():
    def kNNetwork(X_train, X_test, y_train, threshold=False):
        if not threshold:
            threshold = y_train.mean()
        model = Sequential()
        model.add(Dense(units=20, activation='sigmoid', input_dim=13))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        predictions = model.predict(X_test)
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        return predictions

    def KNN(X_train, X_test, y_train, k, threshold=False):
        if not threshold:
            threshold = y_train.mean()
        neigh = KNeighborsRegressor(algorithm='kd_tree', n_neighbors=k)
        neigh.fit(X_train, y_train)
        prediction = neigh.predict(X_test)
        prediction[prediction > threshold] = 1
        prediction[prediction <= threshold] = 0
        return prediction

    def dectree(X_train, X_test, y_train):
        clf = DecisionTreeClassifier(
            random_state=666, min_impurity_decrease=1e-2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def ranforest(X_train, X_test, y_train, threshold=False):
        if not threshold:
            threshold = y_train.mean()
        clf = RandomForestClassifier(
            random_state=666, min_impurity_decrease=1e-4)
        clf.fit(X_train, y_train)
        y_pro = clf.predict_proba(X_test)
        y_pred = np.zeros(len(y_pro))
        for i in range(len(y_pro)):
            if y_pro[i][1] > threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

    def svm(X_train, X_test, y_train, threshold=False):
        if not threshold:
            threshold = y_train.mean()
        clf = SVR(kernel='poly', degree=2, C=1.0, epsilon=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
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
