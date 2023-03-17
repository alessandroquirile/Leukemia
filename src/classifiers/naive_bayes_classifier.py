import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB


def naive_bayes(x, y, cv=5):
    param_grid = {"var_smoothing": [10 ** -9, 10 ** -8, 10 ** -7]}
    grid = GridSearchCV(GaussianNB(), cv=cv, n_jobs=-1, param_grid=param_grid,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_) # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics
