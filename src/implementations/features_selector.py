from interfaces.features_selector import FeaturesSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class FeaturesSelector(FeaturesSelector):

    def __init__(self, model):
        self._model = model

    def select_features(self, df, labels):
        # X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
        # selector = SelectKBest(k=500)
        self._model.fit(df, labels)
        cols = self._model.get_support(indices=True)
        features = df.iloc[:, cols]
        return features

