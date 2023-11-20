from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA #used to reduce dimension
import pandas as pd


def get_linear_regression_model(x_train, x_test, y_train, y_test):
    print("LINEAR MODEL")
    model = LinearRegression()
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_logistic_regression_model(x_train, x_test, y_train, y_test):
    print("LOGISTIC MODEL")
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_random_forest_model(x_train, x_test, y_train, y_test):
    print("RANDOM FOREST")
    model = RandomForestRegressor(random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, np.round(y_pred)))
    print("F1 score: ", metrics.f1_score(y_test, np.round(y_pred), average="weighted"), "\n")
    return model


def get_decision_tree_model(x_train, x_test, y_train, y_test):
    print("DECISION TREE")
    model = DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy:", metrics.accuracy_score( y_test, y_pred))
    print("F1 score: ", metrics.f1_score(y_test, y_pred, average="weighted"), "\n")
    print("\n")
    return model


def calculate_feature_importances(model, columns_to_be_used_as_input):
    importance = model.feature_importances_

    for feature, importance_level in zip(columns_to_be_used_as_input, importance):
        print(f"Feature: {feature}, Importance: {importance_level}")

    fig, ax = plt.subplots()
    plt.bar(x=[x for x in range(len(importance))], height=importance)
    ax.set_xticks(np.arange(len(columns_to_be_used_as_input)))
    ax.set_xticklabels(columns_to_be_used_as_input, rotation=45, rotation_mode='anchor', ha="right")
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()


def calculate_confusion_matrix(model, y_test, y_pred):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()


def clustering(x):
    x_norm = preprocessing.normalize(x)
    model_kmeans = KMeans(n_clusters = 6, random_state = 0, n_init='auto').fit(x_norm)
    pca = PCA(n_components=2) 
    df_2d = pca.fit_transform(x_norm)
    df_new = pd.DataFrame(df_2d, columns=['PC1', 'PC2'])
    df_new['Cluster'] = model_kmeans.labels_
    wcss = []

    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_new)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.show()
