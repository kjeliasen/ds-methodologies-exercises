
###############################################################################
### python imports                                                          ###
###############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from statsmodels.formula.api import ols

from math import sqrt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


###############################################################################
### local imports                                                           ###
###############################################################################

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from dfo import DFO


@timeifdebug
def distance(p, q):
    return math.sqrt((p.petal_length - q.petal_length)**2 +
                     (p.petal_width - q.petal_width)**2)

@timeifdebug
def find_cluster(row: pd.Series):
    distances = centers.apply(lambda center: distance(center, row), axis=1)
    return distances.idxmin()


###############################################################################
### cluster plot functions                                                  ###
###############################################################################

@timeifdebug
def plot_2d_clusters(df, x_col, y_col, c_col, alpha=.05, marker='x', s=1000, c='black', **kwargs):
    sns_colors=sns.color_palette().as_hex()
    centers = df.groupby(c_col).mean()
    clusters=sorted(df[c_col].unique())
#    print(centers[[x_col,y_col]])
    for cluster in clusters:
        subset = df[df[c_col] == cluster]
        plt.scatter(df[x_col], df[y_col], label='cluster '+str(cluster), s=2, marker='o', alpha=alpha, c=sns_colors[cluster])
        plt.text(centers[x_col][cluster], centers[y_col][cluster], cluster, fontsize=24, c=sns_colors[cluster])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    # plt.scatter(centers[x_col], centers[y_col], marker=marker, s=s, c=c, **kwargs)
    # for cluster in clusters:
    #     plt.text(centers[x_col][cluster], centers[y_col][cluster], cluster, fontsize=18)

###############################################################################
### kmeans functions                                                        ###
###############################################################################

@timeifdebug
def compare_ks(df_subset, min_k=1, max_k=10, max_k_pct=.5, **kwargs):
    
    k_values = []
    inertias = []

    df_len = len(df_subset)
    max_len = round(df_len * max_k_pct)

    min_k = 1 if min_k < 1 else min_k
    max_k = max(min_k, min(max_k, max_len))

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k).fit(df_subset)
        inertias.append(kmeans.inertia_)
        k_values.append(k)

    plt.plot(k_values, inertias, marker='x')
    plt.xlabel('K')
    plt.ylabel('inertia')


@timeifdebug
def set_kmeans_clusters(df_subset, n_clusters=5, random_state=12345):
    kmeans = KMeans(n_clusters, random_state=12345).fit(df_subset)
    df_subset['cluster'] = kmeans.labels_
    return df_subset, kmeans



#    sns.relplot(data=df_subset, y=y_vals, x=x_vals, hue='cluster')
#    plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], marker='x', s=2000, c='black')
    