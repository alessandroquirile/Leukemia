import os

import pandas as pd


def create_dataframe(dir1, dir2):
    df1 = _create_dataframe(dir1)
    df2 = _create_dataframe(dir2)
    return pd.concat([df1, df2], ignore_index=True)


def _create_dataframe(dir_path):
    print(dir_path)
    file_names = os.listdir(dir_path)
    full_paths = [os.path.join(dir_path, file_name) for file_name in file_names]
    leukemia = 'leukemia' in dir_path
    dictionary = {'file_name': full_paths, 'leukemia': leukemia}
    return pd.DataFrame(dictionary)