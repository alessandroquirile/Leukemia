import numpy as np
import pandas as pd
from cv2 import SIFT_create
from matplotlib import pyplot as plt
import cv2 as cv
from keras.applications import VGG19, ResNet50, ResNet101
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from classifiers.naive_bayes_classifier import naive_bayes
from classifiers.performance import show_performance_cv, show_performance
from classifiers.multilayer_perceptron import train_deep_neural_network
from dataframe import create_df, get_values, create_full_df, create_features_df
from images import get_image, _show, add_gaussian_noise, crop_image, _crop, _create_mask
from src.factories.features_selector_factory import FeaturesSelectorFactory
from src.factories.features_extractor_factory import FeaturesExtractorFactory
from implementations.sift_extractor import SIFTFeaturesExtractor


if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    images_df = create_df(leukemia_dir, healthy_dir, shuffle=False)  # todo - do shuffle
    labels = get_values(images_df, "leukemia")

    # Feature Extraction
    fe_model = ResNet50(weights='imagenet', include_top=False, pooling="avg")  # Choose your model
    extractor = FeaturesExtractorFactory.get_extractor(fe_model)
    features_df = create_features_df(images_df, extractor=extractor, do_scale=True)
    # print(features_df)

    # Creating whole data frame
    features_and_labels_df = create_full_df(features_df, labels)
    # print(features_and_labels_df)

    # Feature Extraction
    """model = ResNet50(weights='imagenet', include_top=False, pooling="avg")  # Choose your model
    extractor = FeaturesExtractorFactory.get_extractor(model)
    features_df = create_features_df(images_df, extractor=extractor, do_scale=True)
    #print(features_df)"""

    """# Todo - una volta estratte le feature, posso creare un data frame con le feature | labels per la PCA
    full_df = create_full_df(features_df, labels)"""

    """"#Save features to file
    compression_opts = dict(method='zip', archive_name='ResNet50_shuffled_features.csv')
    features_df.to_csv('ResNet50_shuffled_features.zip', index=False, compression=compression_opts)
    """

    # Feature Selection
    """fs_model = SelectKBest(k=50)
    # fs_model = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=30)
    #fs_model = SelectFromModel(
    #    RandomForestClassifier(n_estimators=200, random_state=5, n_jobs=-1),
    #    threshold="1.25*median",
    #    max_features=2
    #)
    selector = FeaturesSelectorFactory.get_features_selector(fs_model)
    features_df = selector.select_features(features_df, labels)

    # ----- DEMO -------
    features_df = pd.read_csv("ResNet101_shuffled_features.zip", index_col=0)
    # print(features_df)"""
