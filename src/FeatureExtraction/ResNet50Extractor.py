import numpy as np
import cv2 as cv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.FeatureExtraction.IFeatureExtractor import IFeatureExtractor


class ResNet50Extractor(IFeatureExtractor):
    def __init__(self):
        # Initializing ResNet50 model
        self._model = ResNet50(weights='imagenet', include_top=False, pooling="avg")

    def extract(self, image) -> np.array:
        img = image.reshape(-1, 224, 224, 3)
        img = preprocess_input(img)

        features = self._model.predict(img)

        return features.reshape(2048, )
