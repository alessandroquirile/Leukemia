import cv2 as cv
import numpy as np

from interfaces.features_extractor import FeaturesExtractor


class SIFTFeaturesExtractor(FeaturesExtractor):
    def __init__(self, model):
        self._model = model

    def extract(self, image) -> np.array:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self._model.detectAndCompute(gray_image, None)
        # image = cv.drawKeypoints(gray_image, keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # _show(image)
        return descriptors
