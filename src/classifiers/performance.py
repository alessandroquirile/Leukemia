import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd

def show_cv_performance(model, scores):
    accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std = \
        get_performance_from_scores(scores)

    print(
        f"Model: {model}\n"
        f"Accuracy: {accuracy_avg} ± {accuracy_std}\n"
        f"Precision: {precision_avg} ± {precision_std}\n"
        f"Recall: {recall_avg} ± {recall_std}\n"
        f"F1: {f1_avg} ± {f1_std}"
    )


def get_performance_from_scores(scores):
    accuracy_avg = scores['mean_test_accuracy'].iloc[0]
    accuracy_std = scores['std_test_accuracy'].iloc[0]
    precision_avg = scores['mean_test_precision'].iloc[0]
    precision_std = scores['std_test_precision'].iloc[0]
    recall_avg = scores['mean_test_recall'].iloc[0]
    recall_std = scores['std_test_recall'].iloc[0]
    f1_avg = scores['mean_test_f1'].iloc[0]
    f1_std = scores['std_test_f1'].iloc[0]

    return accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std


def show_performance(model, y_test, predictions_test):
    accuracy = accuracy_score(y_test, predictions_test)
    precision = precision_score(y_test, predictions_test)
    recall = recall_score(y_test, predictions_test)
    f1 = f1_score(y_test, predictions_test)
    w_f1 = f1_score(y_test, predictions_test, average="weighted")
    print(
        f"Model: {model}\n"
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1: {f1}\n"
        f"Weighted F1: {w_f1}"
    )
    cf_matrix = confusion_matrix(y_test, predictions_test)
    labels = ["Healthy", "Leukemia"]
    sns.heatmap(
        cf_matrix, cmap="PuBu", annot=True, fmt='.0f',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("Ground truth")
    plt.show()
