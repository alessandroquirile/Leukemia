import os

import pandas as pd

from images import crop_image


def create_dataframe(dir1, dir2, shuffle=True):
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


def create_features_dataframe(df, extractor):
    features_list = []
    for row, _ in df.iterrows():
    #for row in range(10):
        cropped_image = crop_image(df, row)
        features = extractor.extract(cropped_image)
        features_list.append(features)
        print(f"\rCurrent: {row}", end='')
    features_df = pd.DataFrame(features_list)
    return features_df
