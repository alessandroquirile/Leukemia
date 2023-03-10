import numpy as np
import pandas as pd
from keras.applications import VGG19, ResNet50, ResNet101
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from classifiers.naive_bayes_classifier import naive_bayes
from classifiers.performance import show_performance_cv, show_performance
from classifiers.multilayer_perceptron import train_deep_neural_network
from dataframe import create_df, get_values, create_features_df
from images import get_image, _show, add_gaussian_noise, crop_image, _crop, _create_mask
from src.factories.features_selector_factory import FeaturesSelectorFactory
from src.factories.features_extractor_factory import FeaturesExtractorFactory

if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_df(leukemia_dir, healthy_dir, shuffle=False)
    labels = get_values(df, "leukemia")

    """
    # Feature Extraction
    model = ResNet50(weights='imagenet', include_top=False, pooling="avg")  # Choose your model
    extractor = FeaturesExtractorFactory.get_extractor(model)
    features_df = create_features_df(df, extractor=extractor, do_scale=True)
    #print(features_df)
    
    #Save features to file
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
    features_df = selector.select_features(features_df, labels)"""

    # ----- DEMO -------
    features_df = pd.read_csv("ResNet101_shuffled_features.zip", index_col=0)
    # print(features_df)



    """
    # PCA Feature Selection
    pca_model = PCA(n_components=500)
    selector = FeaturesSelectorFactory.get_features_selector(pca_model)

    selected_features = selector.select_features(features_df, labels)
    selected_features = pd.DataFrame(selected_features)
    """




    """
    # FS
    fs_model = SelectKBest(k=50)
    selector = FeaturesSelectorFactory.get_features_selector(fs_model)
    features_df = selector.select_features(features_df, labels)
    # print(features_df)
    """

    """
    # Naive bayes Classification
    model, scores = naive_bayes(features_df, labels)  # or knn or svc
    show_performance_cv(model, scores)
    """

    """
    # Neural network classification
    x_train, x_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2)
    model = train_deep_neural_network(x_train, y_train, plot=True)
    model.save("../DNN.h5")

    # if sensitivity is higher, the classifier will be more likely to give a Leukemia result,
    # this enables the user to tune number of False Negatives in favor of False Positives
    prediction_sensitivity = 1

    predictions_test = model.predict(x_test)
    predictions_test = [1 if x >= 0.5 / prediction_sensitivity else 0 for x in predictions_test]
    show_performance(model, y_test, predictions_test)
    """

    image = get_image(df, 2500)
    _show(image, title="Original")

    noisy_image = add_gaussian_noise(image)
    _show(noisy_image, title="Noisy image")

    cropped_image = _crop(noisy_image)
    _show(cropped_image, title="Cropped noisy image")

