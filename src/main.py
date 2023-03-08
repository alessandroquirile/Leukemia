import pandas as pd
from sklearn.feature_selection import SelectKBest

from classifiers.naive_bayes_classifier import naive_bayes
from classifiers.performance import show_performance_cv
from dataframe import create_df, get_values
from src.factories.features_selector_factory import FeaturesSelectorFactory

if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_df(leukemia_dir, healthy_dir, shuffle=False)
    labels = get_values(df, "leukemia")

    # Feature Extraction
    """ model = ResNet101(weights='imagenet', include_top=False, pooling="avg")  # Choose your model
    extractor = factory.get_extractor(model)
    features_df = create_features_df(df, extractor=extractor, do_scale=True)
    print(features_df)"""

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
    features_df = pd.read_csv("ResNet50_unshuffled_features.zip")  # todo - da scalare
    # print(features_df)

    # FS
    fs_model = SelectKBest(k=50)
    selector = FeaturesSelectorFactory.get_features_selector(fs_model)
    features_df = selector.select_features(features_df, labels)
    # print(features_df)

    # Classification
    model, scores = naive_bayes(features_df, labels)  # or knn or svc
    show_performance_cv(model, scores)
