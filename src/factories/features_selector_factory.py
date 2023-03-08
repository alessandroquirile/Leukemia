from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, SelectFromModel

from implementations.features_selector import FeaturesSelector


class FeaturesSelectorFactory:

    @staticmethod
    def get_features_selector(model) -> FeaturesSelector:
        if isinstance(model, SelectKBest) or isinstance(model, RFE) or isinstance(model, SelectFromModel):
            return FeaturesSelector(model)
