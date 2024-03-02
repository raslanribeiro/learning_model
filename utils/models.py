from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
import time

def get_linear_regression_model(x_train, x_test, y_train, y_test):
    print("LINEAR MODEL")
    model = LinearRegression()
    start_time = time.time()
    model.fit(x_train, y_train.values.ravel())
    print(f"elapsed time: {time.time() - start_time}")
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_logistic_regression_model(x_train, x_test, y_train, y_test):
    print("LOGISTIC MODEL")
    model = LogisticRegression(random_state=0)
    start_time = time.time()
    model.fit(x_train, y_train.values.ravel())
    print(f"elapsed time: {time.time() - start_time}")
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_random_forest_model(x_train, x_test, y_train, y_test):
    print("RANDOM FOREST")
    model = RandomForestRegressor(random_state=0)
    start_time = time.time()
    model.fit(x_train, y_train)
    print(f"elapsed time: {time.time() - start_time}")
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_decision_tree_model(x_train, x_test, y_train, y_test):
    print("DECISION TREE")
    model = DecisionTreeClassifier(random_state=0)
    start_time = time.time()
    model.fit(x_train, y_train)
    print(f"elapsed time: {time.time() - start_time}")
    y_pred = model.predict(x_test)
    print("Accuracy:", metrics.accuracy_score( y_test, y_pred))
    print("F1 score: ", metrics.f1_score(y_test, y_pred, average="weighted"), "\n")
    print("\n")
    return model
