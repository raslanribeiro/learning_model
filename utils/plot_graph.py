import matplotlib.pyplot as plt
import numpy as np

def plot_graph(y_pred, y_true):
    plt.figure(figsize=(10,10))
    plt.scatter(y_true.to_numpy(), np.round(y_pred), c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(np.round(y_pred)), max(y_true.to_numpy()))
    p2 = min(min(np.round(y_pred)), min(y_true.to_numpy()))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()