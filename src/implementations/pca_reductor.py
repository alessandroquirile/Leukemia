from src.interfaces.simple_selector import FeaturesSelector


class PCAReductor(FeaturesSelector):

    def __init__(self, model):
        self._model = model

    def select_features(self, df, labels):
        features = self._model.fit_transform(df)
        return features
