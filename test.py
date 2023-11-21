import pandas as pd
from utils import models
from utils.manipulate_data import format_data, encode_columns, columns_to_be_used_as_input, column_to_be_used_as_output
from sklearn.model_selection import train_test_split
from utils.manipulate_data import get_cleaned_data
from sklearn import preprocessing


encoder_instance_storage_type = preprocessing.LabelEncoder()
encoder_photon_acceleration = preprocessing.LabelEncoder()
encoder_runtime_range = preprocessing.LabelEncoder()

# TRAIN MODEL
executions = pd.read_csv("data/to_train/query_executions.csv",thousands=',')
operations = pd.read_csv("data/to_train/query_operations.csv", thousands=',')
instances = pd.read_csv("data/to_train/query_instances.csv", thousands=',')
df = executions.merge(instances, left_on='worker_type', right_on='instance_type', how="left").merge(operations, on='query_name', how="left")
columns_to_be_used_as_input = columns_to_be_used_as_input()
column_to_be_used_as_output = column_to_be_used_as_output()

columns_encoders = [{"column":"instance_storage_type", "encoder":encoder_instance_storage_type},{"column":"photon_acceleration", "encoder":encoder_photon_acceleration},{"column":"runtime_range", "encoder":encoder_runtime_range}]

df = get_cleaned_data(df, columns_encoders, columns_to_be_used_as_input, column_to_be_used_as_output)
x = df[columns_to_be_used_as_input]
y = df[column_to_be_used_as_output]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
model = models.get_decision_tree_model(x_train, x_test, y_train, y_test)



# TEST
data = pd.read_csv("data/to_test/input.csv",thousands=',')
df_formatted = format_data(data)
df_encoded = encode_columns(df_formatted, [{"column":"instance_storage_type", "encoder":encoder_instance_storage_type},{"column":"photon_acceleration", "encoder":encoder_photon_acceleration}])
y_pred = model.predict(df_encoded)
print(encoder_runtime_range.inverse_transform(y_pred))
