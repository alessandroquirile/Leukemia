from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate


def svc(x, y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(x, y)
    model = grid.best_estimator_
    scores = cross_validate(
        model, x, y, cv=5, n_jobs=-1,
        scoring=("accuracy", "precision", "recall", "f1")
    )
    return model, scores
