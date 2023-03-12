from implementations.neural_network_features_extractor import NeuralNetworkFeaturesExtractor
from interfaces.features_extractor import FeaturesExtractor


class FeaturesExtractorFactory:

    @staticmethod
    def get_extractor(model) -> FeaturesExtractor:
        if model.name == "resnet50" or model.name == "resnet101":
            return NeuralNetworkFeaturesExtractor(model, 2048)
        elif model.name == "vgg19":
            return NeuralNetworkFeaturesExtractor(model, 512)
