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
df['runtime_range'] =   np.where(df['runtime'] <= 300, '0 to 300',
                        np.where(df['runtime'] <= 600, '300 to 600',
                        np.where(df['runtime'] <= 1200, '600 to 1200', 
                        np.where(df['runtime'] <= 2400, '1200 to 2400', 
                        np.where(df['runtime'] <= 4800, '2400 to 4800', '> 4800')))))

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


# https://www.myexperiment.org/workflows/4987/versions/1.html
