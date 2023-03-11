from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.decomposition import PCA

from interfaces.simple_selector import FeaturesSelector
from src.implementations.features_selector import SimpleSelector
from src.implementations.pca_reductor import PCAReductor


class FeaturesSelectorFactory:

    @staticmethod
    def get_features_selector(model) -> FeaturesSelector:
        if isinstance(model, SelectKBest) or isinstance(model, RFE) or isinstance(model, SelectFromModel):
            return SimpleSelector(model)
        elif isinstance(model, PCA):
            return PCAReductor(model)
