from abc import abstractmethod


class FeaturesSelector:

    @abstractmethod
    def select_features(self, df, labels):
        pass
