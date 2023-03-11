# This file contains some useful stuff we are going to use soon
# The content of this file has been removed from main.py for improving its readability

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

    # Neural network classification
    """x_train, x_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2)
    model = train_deep_neural_network(x_train, y_train, plot=True)
    model.save("../DNN.h5")

    # if sensitivity is higher, the classifier will be more likely to give a Leukemia result,
    # this enables the user to tune number of False Negatives in favor of False Positives
    prediction_sensitivity = 1

    predictions_test = model.predict(x_test)
    predictions_test = [1 if x >= 0.5 / prediction_sensitivity else 0 for x in predictions_test]
    show_performance(model, y_test, predictions_test)"""
