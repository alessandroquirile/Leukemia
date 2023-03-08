import pandas as pd
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB

from classifiers.knn_classifier import get_best_knn_classifier
from classifiers.performance import show_performance
from dataframe import create_df, get_values, scale
from classifiers.naive_bayes_classifier import get_nb_classifier

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
    x_train, x_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2)

    """
    # SVC
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly']}
    svm_grid = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
    svm_grid.fit(x_train, y_train)
    cls = svm_grid.best_estimator_
    predictions_test = cls.predict(x_test)
    show_performance(cls, y_test, predictions_test)"""

    """
    # NB Classifier
    model = get_nb_classifier(x_train, y_train)
    predictions_test = model.predict(x_test)
    show_performance(model, y_test, predictions_test)"""

    """
    # kNeighbors Classifier
    neighborhood_span = range(5, 25)
    best_model = get_best_knn_classifier(neighborhood_span, x_train, x_test, y_train, y_test, plot=True)
    predictions_test = best_model.predict(x_test)

    show_performance(y_test, predictions_test)"""
