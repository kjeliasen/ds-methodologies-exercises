###############################################################################
### python imports                                                          ###
###############################################################################

# print('Getting Split_Scale', __name__)


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


###############################################################################
### local imports                                                           ###
###############################################################################

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain


###############################################################################
### generic scaling functions                                               ###
###############################################################################

### Test Train Split ##########################################################
# train, test = train_test_split(df, train_size = .80, random_state = 123)


@timeifdebug
def split_my_data_xy(df, target_column, train_pct=.75, random_state=None):
    X = df.drop([target_column], axis=1)
    y = pd.DataFrame(df[target_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=random_state)
    return X_train, X_test, y_train, y_test


@timeifdebug
def split_my_data(df, train_pct=.75, random_state=None):
    train, test = train_test_split(df, train_size=train_pct, random_state=random_state)
    return train, test


### Transform Data ############################################################
@timeifdebug
def scalem(scaler, test, train):
    # transform train
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    # transform test
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled


@timeifdebug
def scale_inverse(train_scaled, test_scaled, scaler):
    # If we wanted to return to original values:
    # apply to train
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    # apply to test
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return train_unscaled, test_unscaled


### Standard Scaler ###########################################################
@timeifdebug
def standard_scaler(train, test):
    # create object & fit
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### Uniform Scaler ############################################################
@timeifdebug
def uniform_scaler(train, test):
    # create scaler object and fit to train
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### Gaussian (Normal) Scaler ##################################################
@timeifdebug
def gaussian_scaler(train, test):
    # create scaler object using yeo-johnson method and fit to train
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled



### MinMax Scaler #############################################################
@timeifdebug
def min_max_scaler(train, test):
    # create scaler object and fit to train
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled



### Robust Scaler #############################################################
@timeifdebug
def iqr_robust_scaler(train, test):
    # create scaler object and fit to train
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


###############################################################################
### project-specific scaling functions                                      ###
###############################################################################



@timeifdebug
def xy_df(dataframe, y_column):
    '''
    xy_df(dataframe, y_column)
    RETURNS X_df, y_df

    Pass in one dataframe of observed data and the name of the target column. Returns dataframe of all columns except the target column and dataframe of just the target column.

    If y_column is a list, more than one column can be separated.
    '''
    X_df = dataframe.drop([y_column], axis=1)
    frame_splain(X_df, title='X')
    y_df = pd.DataFrame(dataframe[y_column])
    frame_splain(y_df, title='y')
    return X_df, y_df


@timeifdebug
def df_join_xy(X, y):
    '''
    df_join_xy(X, y)
    RETURNS dataframe X.join(y)

    Allows reconfigurations of X and y based on train or test and scaled or unscaled    
    '''
    join_df = X.join(y)
    frame_splain(join_df, 'join df')
    return join_df


@timeifdebug
def pairplot_train(dataframe, show_now=True):
    '''
    FUNCTION
    RETURNS:
    '''
    plot = sns.pairplot(dataframe)
    if show_now:
        plt.show()
    else:
        return plot


@timeifdebug
def heatmap_train(dataframe, show_now=True):
    '''
    FUNCTION
    RETURNS:
    '''
    plt.figure(figsize=(7,5))
    cor = dataframe.corr()
    plot = sns.heatmap(cor, annot=True, cmap=plt.cm.RdBu_r)
    if show_now:
        plt.show()
    else:
        return plot


class Context(): pass


@timeifdebug
def set_context(context_df, y_column='taxable_value', train_pct=.75, randomer=None, scaler_fn=standard_scaler):
    '''
    set_context(context_df=get_base_df(), y_column='taxable_value', train_pct=.75, randomer=None, scaler_fn=standard_scaler)
    RETURNS: context object with heaping piles of context enclosed

    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing

    '''
    context = Context()
    context.y_column = y_column
    context.train, context.test = split_my_data(df=context_df, random_state=randomer)
    context.scaler, context.train_scaled, context.test_scaled = scaler_fn(train=context.train, test=context.test)
    context.train['dummy_val']=1
    context.train_scaled['dummy_val']=1
    context.X_train, context.y_train = xy_df(dataframe=context.train, y_column=y_column)
    context.X_test, context.y_test = xy_df(dataframe=context.test, y_column=y_column)
    context.X_train_scaled, context.y_train_scaled = xy_df(dataframe=context.train_scaled, y_column=y_column)
    context.X_test_scaled, context.y_test_scaled = xy_df(dataframe=context.test_scaled, y_column=y_column)
    return context



