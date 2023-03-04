import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_image(df, row):
    file_name = df["file_name"][row]
    image = cv.imread(file_name)
    _show(image, title=file_name)


def _show(image, title=None, cmap=None):
    plt.imshow(image, cmap)
    plt.title(title)
    plt.show()


def crop_image(df, row):
    file_name = df["file_name"][row]
    image = cv.imread(file_name)
    cropped_image = _crop(image)  # no need to use a mask since the images are already pre-processed
    return cropped_image


def _crop(image):
    max_x, max_y, min_x, min_y = _get_blood_cell_coordinates(image)
    return image[min_x:max_x, min_y:max_y, :]


def _get_blood_cell_coordinates(image):
    (x, y, _) = np.where(image != 0)  # relevant image = blood cell and blood cell is not black (0)
    max_x, max_y = np.max((x, y), axis=1)
    min_x, min_y = np.min((x, y), axis=1)
    return max_x, max_y, min_x, min_y


"""""
# No need to use a mask since the images are already pre-processed

def _crop(image):
    mask = _create_mask(image)
    masked_image = cv.bitwise_and(image, image, mask=mask)
    max_x, max_y, min_x, min_y = _get_blood_cell_coordinates(masked_image)
    cropped_image = image[min_x:max_x, min_y:max_y, :]
    return cropped_image


def _create_mask(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_type = cv.THRESH_BINARY + cv.THRESH_OTSU
    mask = cv.threshold(gray_image, 0, 255, thresh_type)[1]
    return mask


def _get_blood_cell_coordinates(masked_image):
    BLACK = 0
    (x, y, _) = np.where(masked_image != BLACK)  # relevant coordinates are those who are not black
    max_x, max_y = np.max((x, y), axis=1)
    min_x, min_y = np.min((x, y), axis=1)
    return max_x, max_y, min_x, min_y
"""
