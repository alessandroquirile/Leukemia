import cv2 as cv
import numpy as np

from images import show_image_
from interfaces.features_extractor import FeaturesExtractor


class SIFTFeaturesExtractor(FeaturesExtractor):
    def __init__(self, model):
        self._model = model

    def extract(self, image) -> np.array:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self._model.detectAndCompute(gray_image, None)
        image = cv.drawKeypoints(gray_image, keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        show_image_(image)
        return descriptors

    def compare(self, image1, image2, n_matches=10):
        keypoints_1, descriptors_1 = self._model.detectAndCompute(image1, None)
        keypoints_2, descriptors_2 = self._model.detectAndCompute(image2, None)

        # feature matching
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:n_matches], image2, flags=2)
        show_image_(img3, title="SIFT Matches")
