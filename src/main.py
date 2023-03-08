from keras.applications import ResNet101

from dataframe import create_df, get_values, create_features_df
from factories.ExtractorFactory import ExtractorFactory
from images import crop_image, _show, _create_mask


if __name__ == '__main__':
    leukemia_dir = "../dataset/leukemia"  # 8491 images
    healthy_dir = "../dataset/healthy"  # 3389 images

    df = create_df(leukemia_dir, healthy_dir, shuffle=False)
    labels = get_values(df, "leukemia")

    factory = ExtractorFactory()
    model = ResNet101(weights='imagenet', include_top=False, pooling="avg")  # Choose your model
    extractor = factory.get_extractor(model)
    features_df = create_features_df(df, extractor=extractor, do_scale=True)
    print(features_df)
