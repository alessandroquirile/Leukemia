import pandas

from FeatureExtraction.SIFT import SIFT
from dataframe import create_dataframe, create_features_dataframe, normalize_features

if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_dataframe(leukemia_dir, healthy_dir, shuffle=False)

    """
    extractor = SIFT()
    features_df = create_features_dataframe(df, extractor=extractor)
    """

    """
    compression_opts = dict(method='zip', archive_name='ResNet50_unshuffled_features.csv')  
    features_df.to_csv('ResNet50_unshuffled_features.zip', index=False, compression=compression_opts)  
    """
    features_df = pandas.read_csv("ResNet50_unshuffled_features.zip")

    print(features_df)
    labels = df["leukemia"].to_numpy()

    features_df = normalize_features(features_df)
    print(features_df)
