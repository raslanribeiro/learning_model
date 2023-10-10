from utils.manipulating_data import get_cleaned_data, columns_to_be_used_as_input, column_to_be_used_as_output
from utils.model_preformance import print_model_performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from utils.functions import mape
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

columns_to_be_used_as_input = columns_to_be_used_as_input()
column_to_be_used_as_output = column_to_be_used_as_output()

df = get_cleaned_data(columns_to_be_used_as_input, column_to_be_used_as_output)

x = df[columns_to_be_used_as_input]
y = df[column_to_be_used_as_output]


# Split data to test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Linear regression
print("LINEAR MODEL")
model = LinearRegression()
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)
print_model_performance(model, y_pred, y_test)
print(f"MAPE: {mape(y_test, y_pred)}")

print("\n------------------------------------\n")

# Logistic regression
print("LOGISTIC MODEL")
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)
print_model_performance(model, y_pred, y_test)
print(f"MAPE: {mape(y_test, y_pred)}")

print("\n------------------------------------\n")

# Decision tree
print("DECISION TREE")
# Add column
df['runtime_range'] =   np.where(df['runtime'] <= 30, '0 to 30', #0 a 30s
                        np.where(df['runtime'] <= 300, '30 to 300', #30s to 5m
                        np.where(df['runtime'] <= 600, '300 to 600', #5m to 10m
                        np.where(df['runtime'] <= 1800, '600 to 1800', #10m to 30m
                        np.where(df['runtime'] <= 3600, '1800 to 3600', #30m to 60m
                        np.where(df['runtime'] <= 7200, '3600 to 7200', #60m to 2h
                        np.where(df['runtime'] <= 14400, '7200 to 14400', '> 14400'))))))) #2h to 4h

# Split data to test again
y = df['runtime_range']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = DecisionTreeClassifier(random_state=0)
dtree = model.fit(x_train, y_train.values.ravel())

plt.figure(figsize=(12,12))
tree.plot_tree(dtree, fontsize=10, feature_names=columns_to_be_used_as_input, class_names=df['runtime_range'], filled=True, proportion=True)
plt.show()

y_pred = dtree.predict(x_test)
print("Accuracy:", metrics.accuracy_score( y_test, y_pred))


# Feature importance
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

#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# Testar clusterização
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA #used to reduce dimension
import pandas as pd

k = 6

x_norm = preprocessing.normalize(x)

model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(x_norm)

pca = PCA(n_components=2) 
df_2d = pca.fit_transform(x_norm)
df_new = pd.DataFrame(df_2d, columns=['PC1', 'PC2'])
df_new['Cluster'] = model.labels_

sb.scatterplot(x="PC1", y="PC2", data = df_new, hue = "Cluster", palette='viridis')
plt.title(f"K-Means Clustering (k={k})")
plt.show()


# ---------------------------------
# Create an empty list to store the WCSS values
wcss = []

# Specify a range of k values to try
k_range = range(1, 11)  # You can adjust the upper limit of k as needed

# Fit K-Means models for each k and calculate WCSS
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_new)  # Replace 'data' with your dataset
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()