import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate


def svc(x, y, cv=5):
    param_grid = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=-1,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_)  # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics
