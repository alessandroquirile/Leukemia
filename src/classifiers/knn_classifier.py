from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier


def knn(x, y, cv=5):
    param_grid = {"n_neighbors": list(range(5, 30))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
    grid.fit(x, y)
    model = grid.best_estimator_
    scores = cross_validate(
        model, x, y, cv=cv, n_jobs=-1,
        scoring=("accuracy", "precision", "recall", "f1")
    )
    return model, scores

