from implementations.neural_network_features_extractor import NeuralNetworkFeaturesExtractor
from interfaces.extractor import Extractor


class ExtractorFactory:

    @staticmethod
    def get_extractor(model) -> Extractor:
        if model.name == "resnet50" or model.name == "resnet101":
            return NeuralNetworkFeaturesExtractor(model, 2048)
        if model.name == "vgg19":
            return NeuralNetworkFeaturesExtractor(model, 512)
        # no handler for SIFT since it's not a model
