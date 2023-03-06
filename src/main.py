from FeatureExtraction.ResNet50Extractor import ResNet50Extractor
from dataframe import create_dataframe, create_features_dataframe

if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_dataframe(leukemia_dir, healthy_dir)

    extractor = ResNet50Extractor()
    features_df = create_features_dataframe(df, extractor=extractor)

    labels = df["leukemia"].to_numpy()
