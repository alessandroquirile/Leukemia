import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input

from src.FeatureExtraction.IFeatureExtractor import IFeatureExtractor


class ResNet50Extractor(IFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._model = ResNet50(weights='imagenet', include_top=False, pooling="avg")

    def extract(self, image) -> np.array:
        size = (224, 224)
        resized_image = cv.resize(image, size)
        reshaped_image = resized_image.reshape(-1, 224, 224, 3)
        reshaped_image = preprocess_input(reshaped_image)
        features = self._model(reshaped_image, training=False)

        return features.numpy().reshape(2048, )
