import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel

from classifiers.naive_bayes_classifier import naive_bayes
from classifiers.performance import show_cv_performance, get_performance_from_scores
from dataframe import read_from_file
from classifiers.knn_classifier import knn
from classifiers.svm import svc
from factories.features_selector_factory import FeaturesSelectorFactory

if __name__ == '__main__':
    # No need to run this if reading from .cvs files
    """leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    images_df = create_df(leukemia_dir, healthy_dir)
    labels = get_values(images_df, "leukemia")"""

    # Reading from zipped .csv files extracted features
    extractor_name = "ResNet101"
    file1 = f"{extractor_name}_features.zip"
    file2 = f"{extractor_name}_labels.zip"
    print(f"Reading from: {file1}, {file2}")
    features_df, labels = read_from_file(file1, file2)

    # Useful for plotting data
    """features_and_labels_df = create_full_df(features_df, labels)
    plot3d(features_and_labels_df)"""

    number_of_features_range = range(50, 500, 50)
    results = []

    for n_features in number_of_features_range:
        print(f"# Number of features: {n_features}")

        k_best_selector = SelectKBest(k=n_features)
        #rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1), n_features_to_select=n_features, verbose=1)
        random_forest_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200, n_jobs=-1),
            threshold="1.25*median",
            max_features=n_features
        )
        pca_selector = PCA(n_features)

        # selection_methods = [k_best_selector, rfe_selector, random_forest_selector, pca_selector]
        selection_methods = [k_best_selector, random_forest_selector, pca_selector]

        for selection_method in selection_methods:
            print(f"\t# Selection method: {selection_method}")

            selector = FeaturesSelectorFactory.get_features_selector(selection_method)
            selected_features_df = selector.select_features(features_df, labels)

            classifiers = [knn, naive_bayes, svc]

            for classifier in classifiers:
                model, scores = classifier(selected_features_df, labels, cv=5)
                #show_cv_performance(model, scores)

                accuracy_avg, accuracy_std, precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std = \
                    get_performance_from_scores(scores)

                result = {"extractor": extractor_name, "n_features": n_features, "selection_method": selection_method,
                          "model": model, "accuracy_avg": accuracy_avg, "accuracy_std": accuracy_std,
                          "precision_avg": precision_avg, "precision_std": precision_std,
                          "recall_avg": recall_avg, "recall_std": recall_std,
                          "f1_avg": f1_avg, "f1_std": f1_std}

                print(f"\t\t{result}")
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{extractor_name}_results.csv")


