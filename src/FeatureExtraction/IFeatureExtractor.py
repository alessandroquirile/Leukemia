from abc import abstractmethod
import numpy as np
import cv2 as cv


class IFeatureExtractor:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract(self, image) -> np.array:
        pass
