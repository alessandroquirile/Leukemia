from matplotlib import pyplot as plt

import numpy as np

def plot_accuracies(neighborhood_span, train_accuracies, test_accuracies, show_best_model=True):
    plt.plot(neighborhood_span, train_accuracies)
    plt.plot(neighborhood_span, test_accuracies, color="orange")
    legends = ["Training Accuracy ", "Testing Accuracy"]
    if show_best_model:
        x_max = neighborhood_span[np.argmax(test_accuracies)]
        y_max = np.max(test_accuracies)
        plt.scatter(x_max, y_max, marker=".", c="red")
        legends.append("Best model")
    plt.legend(legends)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    if type(neighborhood_span) == range:
        plt.xticks(np.arange(min(neighborhood_span), max(neighborhood_span) + 1, 1.0))
    plt.grid(True)
    plt.show()
