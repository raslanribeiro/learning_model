from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

SEED = 42

data = datasets.load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)

dt = DecisionTreeClassifier(max_depth=4,
                            random_state=SEED)
dt.fit(X_train, y_train)


features = data.feature_names
classes = data.target_names
print("teste")