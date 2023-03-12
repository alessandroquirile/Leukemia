from interfaces.simple_selector import FeaturesSelector


class SimpleSelector(FeaturesSelector):

    def __init__(self, model):
        self._model = model

    def select_features(self, df, labels):
        self._model.fit(df, labels)
        cols = self._model.get_support(indices=True)
        features = df.iloc[:, cols]
        return features
