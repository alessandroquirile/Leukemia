
from classifiers.performance import plot_accuracies
from sklearn.naive_bayes import GaussianNB

def get_nb_classifier(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model
