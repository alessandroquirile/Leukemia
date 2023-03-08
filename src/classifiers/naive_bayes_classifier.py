from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB


def naive_bayes(x, y):
    model = GaussianNB()
    scores = cross_validate(
        model, x, y, cv=5, n_jobs=-1,
        scoring=("accuracy", "precision", "recall", "f1")
    )
    return model, scores
