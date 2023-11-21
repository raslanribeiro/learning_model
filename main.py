from utils import models, math
import pandas as pd
from utils.manipulate_data import get_cleaned_data, columns_to_be_used_as_input, column_to_be_used_as_output
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

encoder_instance_storage_type = preprocessing.LabelEncoder()
encoder_photon_acceleration = preprocessing.LabelEncoder()
encoder_runtime_range = preprocessing.LabelEncoder()
columns_encoders = [{"column":"instance_storage_type", "encoder":encoder_instance_storage_type},{"column":"photon_acceleration", "encoder":encoder_photon_acceleration},{"column":"runtime_range", "encoder":encoder_runtime_range}]

# INPUT DATA
executions = pd.read_csv("data/to_train/query_executions.csv",thousands=',')
operations = pd.read_csv("data/to_train/query_operations.csv", thousands=',')
instances = pd.read_csv("data/to_train/query_instances.csv", thousands=',')

# MERGE DATA
df = executions.merge(instances, left_on='worker_type', right_on='instance_type', how="left").merge(operations, on='query_name', how="left")

columns_to_be_used_as_input = columns_to_be_used_as_input()
column_to_be_used_as_output = column_to_be_used_as_output()
df = get_cleaned_data(df, columns_encoders, columns_to_be_used_as_input, column_to_be_used_as_output)
x = df[columns_to_be_used_as_input]
y = df[column_to_be_used_as_output]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

models.get_linear_regression_model(x_train, x_test, y_train, y_test)
models.get_logistic_regression_model(x_train, x_test, y_train, y_test)
models.get_random_forest_model(x_train, x_test, y_train, y_test)
model = models.get_decision_tree_model(x_train, x_test, y_train, y_test)
math.calculate_feature_importances(model, columns_to_be_used_as_input)
math.calculate_confusion_matrix(model, encoder_runtime_range, y_test, model.predict(x_test))
math.clustering(x)