from implementations.neural_network_features_extractor import NeuralNetworkFeaturesFeaturesExtractor
from interfaces.features_extractor import FeaturesExtractor


class FeaturesExtractorFactory:

    @staticmethod
    def get_extractor(model) -> FeaturesExtractor:
        if model.name == "resnet50" or model.name == "resnet101":
            return NeuralNetworkFeaturesFeaturesExtractor(model, 2048)
        if model.name == "vgg19":
            return NeuralNetworkFeaturesFeaturesExtractor(model, 512)
        # no handler for SIFT since it's not a model