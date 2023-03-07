import cv2 as cv
import numpy as np
from keras.applications.resnet import preprocess_input

from interfaces.Extractor import Extractor


class VGG19Extractor(Extractor):
    def __init__(self, model):
        self._model = model

    def extract(self, image) -> np.ndarray:
        size = (224, 224)
        resized_image = cv.resize(image, size)
        reshaped_image = resized_image.reshape(-1, *size, 3)
        reshaped_image = preprocess_input(reshaped_image)
        features = self._model(reshaped_image, training=False)
        features = features.numpy().reshape(512, )
        return features
