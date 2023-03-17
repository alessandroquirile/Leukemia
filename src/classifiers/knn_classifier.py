import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier


def knn(x, y, cv=5):
    param_grid = {"n_neighbors": list(range(5, 30))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, n_jobs=-1,
                        scoring=("accuracy", "precision", "recall", "f1"),
                        refit="f1")
    grid.fit(x, y)

    best_model = grid.best_estimator_
    results = pd.DataFrame(grid.cv_results_) # results contains all the metrics for every parameter combination
    best_model_metrics = results.iloc[[grid.best_index_]]  # select only the row of the best classifier
    return best_model, best_model_metrics
