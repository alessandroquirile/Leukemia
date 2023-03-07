from implementations.ResNet101Extractor import ResNet101Extractor
from implementations.ResNet50Extractor import ResNet50Extractor
from implementations.VGG19Extractor import VGG19Extractor
from interfaces.Extractor import Extractor


class ExtractorFactory:
    _instance = None

    def __init__(self):
        self._instance = None

    @staticmethod
    def get_instance():
        if ExtractorFactory._instance is None:
            ExtractorFactory._instance = ExtractorFactory()
        return ExtractorFactory._instance

    @staticmethod
    def get_extractor(model) -> Extractor:
        if model.name == "resnet50":
            return ResNet50Extractor(model)
        if model.name == "resnet101":
            return ResNet101Extractor(model)
        if model.name == "vgg19":
            return VGG19Extractor(model)
        # no handler for SIFT since it's not a model
