from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import numpy as np
from math import sqrt

def print_model_performance(model, y_pred, y_true):
    # print('Coefficients:', model.coef_)
    # print('Intercept:', model.intercept_)
    print('Coefficient of determination (R^2): %.3f' % r2_score(y_true, y_pred))
    print('Mean squared error (MSE): %.3f'% mean_squared_error(y_true, y_pred))
    print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_true, y_pred)) )
    # print("Accuracy: ", accuracy_score(y_true, np.round(y_pred)))
    # print("F1 score: ", f1_score(y_true, np.round(y_pred), average="micro"))