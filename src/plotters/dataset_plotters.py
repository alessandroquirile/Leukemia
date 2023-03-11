from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def plot3d(features_and_labels_df):
    features = features_and_labels_df.drop('leukemia', axis=1).values
    labels = features_and_labels_df['leukemia'].values
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)
    df_pca = pd.DataFrame(data=features_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['leukemia'] = labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_pca.loc[df_pca['leukemia'] == True, 'PC1'], df_pca.loc[df_pca['leukemia'] == True, 'PC2'],
               df_pca.loc[df_pca['leukemia'] == True, 'PC3'], color='red', label='leukemia')
    ax.scatter(df_pca.loc[df_pca['leukemia'] == False, 'PC1'], df_pca.loc[df_pca['leukemia'] == False, 'PC2'],
               df_pca.loc[df_pca['leukemia'] == False, 'PC3'], color='blue', label='healthy')
    ax.legend()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()


def plot2d(features_and_labels_df):
    features = features_and_labels_df.drop('leukemia', axis=1).values
    labels = features_and_labels_df['leukemia'].values
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    df_pca = pd.DataFrame(data=features_pca, columns=['PC1', 'PC2'])
    df_pca['leukemia'] = labels
    plt.scatter(df_pca.loc[df_pca['leukemia'] == True, 'PC1'], df_pca.loc[df_pca['leukemia'] == True, 'PC2'],
                color='red', label='leukemia=True')
    plt.scatter(df_pca.loc[df_pca['leukemia'] == False, 'PC1'], df_pca.loc[df_pca['leukemia'] == False, 'PC2'],
                color='blue', label='leukemia=False')
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
