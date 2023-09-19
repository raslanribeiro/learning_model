from sklearn.ensemble import RandomForestRegressor
from utils.manipulating_data import get_cleaned_data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from utils.model_preformance import print_model_performance


df = get_cleaned_data()

x = df[["input_rows_quantity","output_rows_quantity","number_of_workers", "photon_acceleration", "constraint","cte","case_when","inner_join","left_join","right_join","group_by","where","and","or","subquery", "vcpu", "memory_ram_gb", "instance_storage_type"]]
y = df[["runtime"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(x_train, y_train)


print('Random Forest R squared: %.3f' % forest_reg.score(x_test, y_test))

y_pred = forest_reg.predict(x_test)
forest_mse = mean_squared_error(y_test, y_pred)
forest_rmse = np.sqrt(forest_mse)
print('Random Forest RMSE: %.3f' % forest_rmse)
print("Accuracy: ", accuracy_score(y_test, np.round(y_pred)))
print("F1 score: ", f1_score(y_test, np.round(y_pred), average="micro"))
