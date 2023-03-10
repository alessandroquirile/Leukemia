from sklearn.decomposition import PCA
from src.interfaces.features_selector import FeaturesSelector


class PCASelector(FeaturesSelector):

    def __init__(self, model):
        self._model = model

    def select_features(self, df, labels):
        features = self._model.fit_transform(df)
        return features
