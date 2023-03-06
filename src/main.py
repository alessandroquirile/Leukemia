import os

from dataframe import create_dataframe
from images import show_image, crop_image
import cv2 as cv
import matplotlib.pyplot as plt
from FeatureExtraction.ResNet50Extractor import ResNet50Extractor

if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_dataframe(leukemia_dir, healthy_dir)
    print(df)

    show_image(df, 1000)

    file_name = df["file_name"][1000]
    image = cv.imread(file_name)
    extractor = ResNet50Extractor()
    extractor.extract(image)
    # crop_image(df, 1000)
