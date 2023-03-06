from abc import abstractmethod

import numpy as np


class IFeatureExtractor:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract(self, image) -> np.array:
        pass
