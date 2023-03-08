from implementations.resnet101_extractor import ResNet101Extractor
from implementations.resnet50_extractor import ResNet50Extractor
from implementations.vgg19_extractor import VGG19Extractor
from interfaces.extractor import Extractor


class ExtractorFactory:

    @staticmethod
    def get_extractor(model) -> Extractor:
        if model.name == "resnet50":
            return ResNet50Extractor(model)
        if model.name == "resnet101":
            return ResNet101Extractor(model)
        if model.name == "vgg19":
            return VGG19Extractor(model)
        # no handler for SIFT since it's not a model
