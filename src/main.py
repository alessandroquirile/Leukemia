from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction.ResNet50Extractor import ResNet50Extractor
from dataframe import create_dataframe, create_features_dataframe


if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_dataframe(leukemia_dir, healthy_dir, shuffle=False)

    extractor = ResNet50Extractor()
    features_df = create_features_dataframe(df, extractor=extractor)
    features_df.to_csv("ResNet50_unshuffled_features.csv")

    labels = df["leukemia"].to_numpy()

    print(features_df)
    print(labels)

    features_df = normalize_features(features_df)
    print(features_df)

