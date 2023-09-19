import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# def mape(y_test, y_pred):
#     y_test, y_pred = np.array(y_test), np.array(y_pred)
#     mape = np.mean(np.abs((y_test - y_pred) / y_test))
#     return mape

def mape(y_test, y_pred):
    return mean_absolute_percentage_error(y_test, y_pred)