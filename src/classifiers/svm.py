from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate


def svc(x, y, cv=5):
    param_grid = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['auto']}
    grid = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
    grid.fit(x, y)
    model = grid.best_estimator_
    scores = cross_validate(
        model, x, y, cv=cv, n_jobs=-1,
        scoring=("accuracy", "precision", "recall", "f1")
    )
    return model, scores
