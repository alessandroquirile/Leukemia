from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB


def naive_bayes(x, y, cv=5):
    param_grid = {"var_smoothing": [10 ** -9, 10 ** -8, 10 ** -7]}
    grid = GridSearchCV(GaussianNB(), cv=cv, n_jobs=-1, param_grid=param_grid)
    grid.fit(x, y)
    model = grid.best_estimator_
    scores = cross_validate(
        model, x, y, cv=cv, n_jobs=-1,
        scoring=("accuracy", "precision", "recall", "f1")
    )
    return model, scores
