from sklearn.feature_selection import SelectKBest

from classifiers.naive_bayes_classifier import naive_bayes
from classifiers.performance import show_cv_performance
from dataframe import read_from_file
from src.factories.features_selector_factory import FeaturesSelectorFactory

if __name__ == '__main__':
    # No need to run this if reading from .cvs files
    """leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    images_df = create_df(leukemia_dir, healthy_dir)
    labels = get_values(images_df, "leukemia")"""

    # Reading from zipped .csv files extracted features
    file1 = "ResNet50_features.zip"
    file2 = "ResNet50_labels.zip"
    print(f"Reading from: {file1}, {file2}")
    features_df, labels = read_from_file(file1, file2)

    # Useful for plotting data
    """features_and_labels_df = create_full_df(features_df, labels)
    plot3d(features_and_labels_df)"""

    # Feature selection
    fs_model = SelectKBest(k=50)
    selector = FeaturesSelectorFactory.get_features_selector(fs_model)
    features_df = selector.select_features(features_df, labels)

    # Classification
    model, scores = naive_bayes(features_df, labels, cv=5)
    show_cv_performance(model, scores)

    # Todo: sklearn.exceptions.NotFittedError: This GaussianNB instance is not fitted yet.
    #       Call 'fit' with appropriate arguments before using this estimator.
    model.predict(features_df)
