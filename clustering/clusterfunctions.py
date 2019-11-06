
###############################################################################
### python imports                                                          ###
###############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def distance(p, q):
    return math.sqrt((p.petal_length - q.petal_length)**2 +
                     (p.petal_width - q.petal_width)**2)

def find_cluster(row: pd.Series):
    distances = centers.apply(lambda center: distance(center, row), axis=1)
    return distances.idxmin()


def plot_2d_clusters(df, x_col, y_col, c_col, alpha=.05, marker='x', s=1000, c='black', **kwargs):
    centers = df.groupby(c_col).mean()
    for cluster in df[c_col].unique():
        subset = df[df[c_col] == cluster]
        plt.scatter(df[x_col], df[y_col], label=cluster, alpha=alpha)
    plt.legend()
    plt.scatter(centers[x_col], centers[y_col], marker=marker, s=s, c=c)


###############################################################################
### kmeans functions                                                        ###
###############################################################################

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


def set_kmeans_clusters(df_subset, n_clusters=5):
    kmeans = KMeans(n_clusters).fit(df_subset)
    df_subset['cluster'] = kmeans.labels_
    return df_subset



#    sns.relplot(data=df_subset, y=y_vals, x=x_vals, hue='cluster')
#    plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], marker='x', s=2000, c='black')
    