import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def show_cv_performance(model, scores):
    accuracies = scores["test_accuracy"]
    accuracy_avg = accuracies.mean()
    accuracy_std = accuracies.std()
    precisions = scores["test_precision"]
    precision_avg = precisions.mean()
    precision_std = precisions.std()
    recalls = scores["test_recall"]
    recall_avg = recalls.mean()
    recall_std = recalls.std()
    f1s = scores["test_f1"]
    f1_avg = f1s.mean()
    f1_std = f1s.std()
    print(
        f"Model: {model}\n"
        f"Accuracy: {accuracy_avg} ± {accuracy_std}\n"
        f"Precision: {precision_avg} ± {precision_std}\n"
        f"Recall: {recall_avg} ± {recall_std}\n"
        f"F1: {f1_avg} ± {f1_std}"
    )


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
