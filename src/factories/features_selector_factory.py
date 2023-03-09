from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, SelectFromModel

from implementations.features_selector import Selector


class FeaturesSelectorFactory:

    @staticmethod
    def get_features_selector(model) -> Selector:
        if isinstance(model, SelectKBest) or isinstance(model, RFE) or isinstance(model, SelectFromModel):
            return Selector(model)
