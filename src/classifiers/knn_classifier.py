from sklearn.neighbors import KNeighborsClassifier

from classifiers.performance import plot_accuracies


def get_best_knn_classifier(neighborhood_span, x_train, x_test, y_train, y_test, plot=False):
    train_accuracies = []
    test_accuracies = []
    best_acc = 0
    best_model = None
    for neighborhood_size in neighborhood_span:
        knn = KNeighborsClassifier(neighborhood_size, n_jobs=-1)
        knn.fit(x_train, y_train)
        train_accuracy = knn.score(x_train, y_train)
        test_accuracy = knn.score(x_test, y_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if best_acc < test_accuracy:
            best_acc = test_accuracy
            best_model = knn
    if plot:
        plot_accuracies(neighborhood_span, train_accuracies, test_accuracies)

    return best_model
