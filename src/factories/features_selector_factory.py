from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.decomposition import PCA

from src.implementations.features_selector import Selector
from src.implementations.pca_selector import PCASelector


class FeaturesSelectorFactory:

    @staticmethod
    def get_features_selector(model) -> Selector:
        if isinstance(model, SelectKBest) or isinstance(model, RFE) or isinstance(model, SelectFromModel):
            return Selector(model)
        elif isinstance(model, PCA):
            return PCASelector(model)
