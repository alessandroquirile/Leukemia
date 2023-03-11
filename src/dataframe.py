import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from images import crop_image


def create_df(dir1, dir2, shuffle=True):
    df1 = _create_dataframe(dir1)
    df2 = _create_dataframe(dir2)
    df = pd.concat([df1, df2], ignore_index=True)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def _create_dataframe(dir_path):
    file_names = os.listdir(dir_path)
    full_paths = [os.path.join(dir_path, file_name) for file_name in file_names]
    leukemia = 'leukemia' in dir_path
    dictionary = {'file_name': full_paths, 'leukemia': leukemia}
    return pd.DataFrame(dictionary)


def create_features_df(df, extractor, do_scale):
    features_list = []
    for row in range(len(df)):
    #for row in range(10):  # dbg
        cropped_image = crop_image(df, row)
        features = extractor.extract(cropped_image)
        features_list.append(features)
        print(f"\rCurrent: {row}/{len(df)}", end='')
    features_df = pd.DataFrame(features_list)
    if do_scale:
        features_df = scale(features_df)
    return features_df


def get_values(df, column):
    return df[column].to_numpy()


def scale(features_df):
    scaler = MinMaxScaler()
    scaler.fit(features_df)
    scaled_features = scaler.transform(features_df)
    return pd.DataFrame(scaled_features)


def create_full_df(features_df, labels):
    labels_df = pd.DataFrame({'leukemia': labels})
    return pd.concat([features_df, labels_df], axis=1)