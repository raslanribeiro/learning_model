from utils import models, math
import pandas as pd
from utils.manipulate_data import get_cleaned_data, columns_to_be_used_as_input, column_to_be_used_as_output, generate_synthetic_data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import numpy as np

def choose_model(x_train, x_test, y_train, y_test):
    print("Model 1:")
    model1 = models.get_linear_regression_model(x_train, x_test, y_train, y_test)

    print("Model 2:")
    model2 = models.get_logistic_regression_model(x_train, x_test, y_train, y_test)

    print("Model 3:")
    model3 = models.get_random_forest_model(x_train, x_test, y_train, y_test)

    print("Model 4:")
    model4 = models.get_decision_tree_model(x_train, x_test, y_train, y_test)

    model_number = int(input("Enter model's number to proceed: "))
    while True:
        if model_number in [1, 2, 3, 4]:
            break

    return eval(f"model{model_number}")

def app():
    encoder_instance_storage_type = preprocessing.LabelEncoder()
    encoder_photon_acceleration = preprocessing.LabelEncoder()
    encoder_runtime_range = preprocessing.LabelEncoder()
    columns_encoders = [{"column":"instance_storage_type", "encoder":encoder_instance_storage_type},{"column":"photon_acceleration", "encoder":encoder_photon_acceleration},{"column":"runtime_range", "encoder":encoder_runtime_range}]

    # INPUT DATA
    executions = pd.read_csv("data/to_train/query_executions.csv",thousands=',')
    operations = pd.read_csv("data/to_train/query_operations.csv", thousands=',')
    instances = pd.read_csv("data/to_train/query_instances.csv", thousands=',')

    # MERGE DATA
    df_initial = executions.merge(instances, left_on='worker_type', right_on='instance_type', how="left").merge(operations, on='query_name', how="left")

    input_columns = columns_to_be_used_as_input()
    output_columns = column_to_be_used_as_output()

    df = get_cleaned_data(df_initial, columns_encoders, input_columns, output_columns, 6)

    x = df[input_columns]
    y = df[output_columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    model = choose_model(x_train, x_test, y_train, y_test)

    ##################################Increasing sample data#######################################
    final_number_of_rows = 1000
    df_synthetic = generate_synthetic_data(df_initial, final_number_of_rows)
    noise = np.random.uniform(low=-0.3, high=0.3, size=len(df_synthetic))
    df_synthetic["runtime"] = df_synthetic["runtime"] + (df_synthetic["runtime"] * noise)
    df_synthetic = get_cleaned_data(df_synthetic, columns_encoders, input_columns, output_columns, 6)
    df_synthetic["runtime_range"] = model.predict(df_synthetic.drop(columns=["runtime_range"]))

    final_df = pd.concat([df, df_synthetic], ignore_index=True)

    x_new = final_df[input_columns]
    y_new = final_df[output_columns]
    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.25, random_state=0)
    model = choose_model(x_train, x_test, y_train, y_test)
    ################################################################################################ 

    math.calculate_feature_importances(model, input_columns)
    math.calculate_confusion_matrix(model, encoder_runtime_range, y_test, model.predict(x_test))
    math.clustering(x)


if __name__ == "__main__":
    app()