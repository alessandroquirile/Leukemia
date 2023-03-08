from interfaces.features_selector import FeaturesSelector
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_classif
from implementations.features_selector import FeaturesSelector
from sklearn.feature_selection import RFE

class FeaturesSelectorFactory:

    @staticmethod
    def get_features_selector(model) -> FeaturesSelector:
        if isinstance(model, SelectKBest) or isinstance(model, RFE) or isinstance(model, SelectFromModel):
            return FeaturesSelector(model)
