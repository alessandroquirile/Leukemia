from abc import abstractmethod


class Extractor:

    @abstractmethod
    def extract(self, image):
        pass
