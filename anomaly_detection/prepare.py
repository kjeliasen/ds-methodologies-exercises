# Exercises
# The end product of this exercise should be the specified functions in a 
# python script named prepare.py. Do these in your 
# classification_exercises.ipynb first, then transfer to the prepare.py file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


import acquire as acq

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from dfo import DFO, set_dfo


###############################################################################
### preparation functions                                                   ###
###############################################################################

@timeifdebug
def is_in_both():
    pass


@timeifdebug
def encode_col(df, col, **kwargs):
    '''
    encode_col(df, col, **kwargs)
    RETURNS: df, encoder
    '''
    encoder = LabelEncoder()
    encoder.fit (df[[col]])
    df[[col]] = encoder.transform(df[[col]])
    return df, encoder


@timeifdebug
def simpute(df, col, missing_values=np.nan, strategy='most_frequent', splain=local_settings.splain, **kwargs):
    '''
    simpute(df, column, missing_values=np.nan, strategy='most_frequent', splain=local_settings.splain, **kwargs)
    RETURNS: df
    '''
    df[[col]] = df[[col]].fillna(missing_values)
    imp_mode = SimpleImputer(missing_values=missing_values, strategy=strategy)
    df[[col]] = imp_mode.fit_transform(df[[col]])
    return df


@timeifdebug
def retype_cols(df, cols, to_dtype, **kwargs):
    '''
    retype_cols(df, columns, to_dtype, **kwargs)
    RETURNS df with updated column types
    
    Function first checks to ensure columns are in dataframe.
    '''
    for col in (xcol for xcol in cols if xcol in df.columns):
        df[col] = df[col].astype(to_dtype)
    return df


@timeifdebug
def remove_cols(df, cols, **kwargs):
    '''
    drop_cols(df, cols, **kwargs)
    RETURNS df with cols removed
    
    Function first checks to ensure columns are in dataframe.
    '''
    dropcols = [col for col in cols if col in df.columns]
    if len(dropcols):
        df = df.drop(columns=dropcols)    
    return df
    

@timeifdebug
def convert_to_dates(df, cols=[], format='%Y-%m-%d', errors='coerce', **kwargs):
    '''
    convert_to_dates(
        df, 
        cols, 
        format='%y-%m-%d', 
        errors='coerce', 
        **kwargs
    )
    RETURNS df with columns updated to dates
    '''
    convertcols = [col for col in cols if col in df.columns]
    for col in convertcols:
        df[col] = pd.to_datetime(df[col], errors=errors, format=format)
    return df    


@timeifdebug
def nulls_by_row(df):
    total_cols = df.shape[1]
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/ total_cols
    rows_missing = (
        pd.DataFrame(
        {
            'num_cols_missing': num_cols_missing, 
            'pct_cols_missing': pct_cols_missing
        }
        )
        .reset_index()
        .groupby(['num_cols_missing','pct_cols_missing'])
        .count()
        .rename(index=str, columns={'index': 'num_rows'})
        .reset_index()
    )
    return rows_missing 

###############################################################################
### split-scale functions                                                   ###
###############################################################################

### Test Train Split ##########################################################
# train, test = train_test_split(df, train_size = .80, random_state = 123)
@timeifdebug
def split_my_data(df, y_column, train_pct=.75, stratify=None, random_state=None, **kwargs):
    X = df.drop([y_column], axis=1)
    y = pd.DataFrame(df[y_column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=random_state)
    return X_train, X_test, y_train, y_test


@timeifdebug
def split_my_data_whole(df, train_pct=.75, stratify=None, random_state=None, **kwargs):
    train, test = train_test_split(df, train_size=train_pct, random_state=random_state)
    return train, test


### Transform Data ############################################################
@timeifdebug
def scalem(scaler, train, test, **kwargs):
    # transform train
    scaler.fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    # transform test
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled


@timeifdebug
def scale_inverse(train_scaled, test_scaled, scaler, **kwargs):
    # If we wanted to return to original values:
    # apply to train
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    # apply to test
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return train_unscaled, test_unscaled


###############################################################################
### scaler creation functions                                               ###
###############################################################################

### Standard Scaler ###########################################################
@timeifdebug
def standard_scaler(train, test, copy=True, with_mean=True, with_std=True, **kwargs):
    # create object & fit
    scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### Uniform Scaler ############################################################
@timeifdebug
def uniform_scaler(train, test, n_quantiles=100, output_distribution='uniform', random_state=123, copy=True, **kwargs):
    # create scaler object and fit to train
    scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=random_state, copy=True).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### Gaussian (Normal) Scaler ##################################################
@timeifdebug
def gaussian_scaler(train, test, method='yeo-johnson', standardize=False, copy=True, **kwargs):
    # create scaler object using yeo-johnson method and fit to train
    scaler = PowerTransformer(method=method, standardize=standardize, copy=copy).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### MinMax Scaler #############################################################
@timeifdebug
def min_max_scaler(train, test, copy=True, feature_range=(0,1), **kwargs):
    # create scaler object and fit to train
    scaler = MinMaxScaler(copy=copy, feature_range=feature_range).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


### Robust Scaler #############################################################
@timeifdebug
def iqr_robust_scaler(train, test, quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True, **kwargs):
    # create scaler object and fit to train
    scaler = RobustScaler(quantile_range=quantile_range, copy=copy, with_centering=with_centering, with_scaling=with_scaling).fit(train)
    # scale'm
    train_scaled, test_scaled = scalem(scaler=scaler, test=test, train=train)
    return scaler, train_scaled, test_scaled


###############################################################################
### split/merge functions                                                   ###
###############################################################################

### X y splits ################################################################
@timeifdebug
def xy_df(df, y_column):
    '''
    xy_df(
        df, 
        y_column
    )
    RETURNS X_df, y_df

    Pass in one dataframe of observed data and the name of the target column. Returns dataframe of all columns except the target column and dataframe of just the target column.

    If y_column is a list, more than one column can be separated.
    '''
    X_df = df.drop([y_column], axis=1)
    frame_splain(X_df, title='X')
    y_df = pd.DataFrame(df[y_column])
    frame_splain(y_df, title='y')
    return X_df, y_df


@timeifdebug
def df_join_xy(X, y):
    '''
    df_join_xy(
        X, 
        y
    )
    RETURNS dataframe X.join(y)

    Allows reconfigurations of X and y based on train or test and scaled or unscaled    
    '''
    join_df = X.join(y)
    frame_splain(join_df, 'join df')
    return join_df


###############################################################################
### plotting functions                                                      ###
###############################################################################

@timeifdebug
def pairplot_train(df, show_now=True):
    '''
    pairplot_train(
        df, 
        show_now=True
    )
    RETURNS:
    '''
    plot = sns.pairplot(dataframe)
    if show_now:
        plt.show()
    else:
        return plot


@timeifdebug
def heatmap_train(df, show_now=True):
    '''
    heatmap_train(
        df, 
        show_now=True
    )
    RETURNS:
    '''
    plt.figure(figsize=(7,5))
    cor = dataframe.corr()
    plot = sns.heatmap(cor, annot=True, cmap=plt.cm.RdBu_r)
    if show_now:
        plt.show()
    else:
        return plot




###############################################################################
### DFO functions                                                           ###
###############################################################################

@timeifdebug
def split_dfo(dfo, train_pct=.7, randomer=None, stratify=None, drop_cols=None, splain=local_settings.splain, **kwargs):
    '''
    scale_dfo(dfo, scaler_fn=standard_scaler, **kwargs)
    RETURNS: dfo object with heaping piles of context enclosed
    
    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing
    '''
    dfo.randomer = randomer
    dfo.stratify = stratify if stratify is not None else dfo.y_column
    dfo.train_pct = train_pct
    dfo.drop_cols = drop_cols
    df2 = pd.DataFrame(dfo.df)
    df2 = remove_cols(df=df2, cols=drop_cols)
    dfo.train, dfo.test = split_my_data_whole(df=df2, target_column=dfo.y_column, stratify=dfo.stratify, random_state=dfo.randomer)
    dfo.train_index = dfo.train.index
    frame_splain(dfo.train, 'DFO Train', splain=splain)
    dfo.test_index = dfo.test.index
    frame_splain(dfo.test, 'DFO Test', splain=splain)
    return dfo


@timeifdebug
def scale_dfo(dfo, scaler_fn=standard_scaler, splain=local_settings.splain, **kwargs):
    '''
    scale_dfo(dfo, scaler_fn=standard_scaler, **kwargs)
    RETURNS: dfo object with heaping piles of context enclosed
    
    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing
    '''

    dfo.scaler_fn = scaler_fn
    if scaler_fn is None:
        dfo.scaler = None
    else:
        dfo.scaler, dfo.train_scaled, dfo.test_scaled = scaler_fn(train=dfo.train, test=dfo.test)
        dfo.train_scaled['dummy_val']=1
        dfo.test_scaled['dummy_val']=1
    dfo.train['dummy_val']=1
    dfo.test['dummy_val']=1
    dfo.X_train, dfo.y_train = xy_df(dataframe=dfo.train, y_column=dfo.y_column)
    dfo.X_test, dfo.y_test = xy_df(dataframe=dfo.test, y_column=dfo.y_column)
    frame_splain(dfo.X_train, 'X_Train', splain=splain)
    frame_splain(dfo.y_train, 'y_Train', splain=splain)
    frame_splain(dfo.X_test, 'X_Test', splain=splain)
    frame_splain(dfo.y_test, 'Y_Test', splain=splain)
    if scaler_fn is not None:
        dfo.X_train_scaled, dfo.y_train_scaled = xy_df(dataframe=dfo.train_scaled, y_column=dfo.y_column)
        dfo.X_test_scaled, dfo.y_test_scaled = xy_df(dataframe=dfo.test_scaled, y_column=dfo.y_column)
        frame_splain(dfo.X_train_scaled, 'X_Train_scaled', splain=splain)
        frame_splain(dfo.y_train_scaled, 'y_Train_scaled', splain=splain)
        frame_splain(dfo.X_test_scaled, 'X_Test_scaled', splain=splain)
        frame_splain(dfo.y_test_scaled, 'Y_Test_scaled', splain=splain)
    
    return dfo


###############################################################################
### value analysis functions                                                ###
###############################################################################

@timeifdebug
def get_column_values_stats(
        df, 
        get_cols=[], 
        max_uniques=10, 
        target_col='',
        limit_to_max=False,
        allow_zero_cols=False,
        **kwargs
        ):
    '''
    get_column_values_stats(
        df, 
        get_cols=[], 
        max_uniques=10, 
        target_col='', 
        limit_to_max=False,
        allow_zero_cols=False,
        **kwargs
        )
    RETURNS summary dataframe

    Receives dataframe as input, examines all columns defined as objects, and
    returns a summary report with column name as its index.

    Report shows the following information about the columns of the dataframe:
        cols: column name
        dtypes: data type
        num_rows_values: number of rows with non-null values 
        num_rows_missing: number of rows with missing values
        pct_rows_missing: percentage of rows with missing values
        num_uniques: number of unique values 
        unique_values: list of unique values if the unique value count is less 
            than or equal to max_uniques.
        
    If the input dataframe contains the target column, enter that name as the 
    target_col argument and it will be removed from the analysis.
    
    If limit_to_max is True (default), the resulting dataframe will only show
    columns with no more unique values than specified in max_uniques.
    
    If allow_zero_cols is True, the resulting dataframe will return no rows if 
        there are no columns found meeting the selection criteria. If False 
        (default), all columns from the input dataframe will be returned.


    NOTE: This function can be used to identify and then re-type catergorical
    columns. Run the function with the following parameters, change the data 
    types of know categorical values to 'category', then re-run this function, 
    gradually increasing the max_uniques value until the desired limit for 
    categoricals has been met.
    
    get_column_values_stats(
        df, 
        get_cols=df.columns[df.dtypes!='category'],
        max_uniques=5,
        limit_to_max=True
        )
    '''
    # get rowcount
    num_recs = len(df)
    
    # ensure all use_cols are actually present in df
    use_cols = [col for col in get_cols if col in df.columns]
    # if no columns presented, get all columns from df
    cols = df.columns if len(use_cols) == 0 and allow_zero_cols==False else get_cols
    
    # make df for all columns, exclude target column if passed
    df_cols = pd.DataFrame(cols, columns=['cols'])
    df_cols = df_cols[df_cols.cols != target_col]
    
    # get statistics on all columns
    df_cols['dtype'] = df_cols.cols.apply(lambda x: df[x].dtype)
    df_cols['num_rows_values'] = df_cols.cols.apply(lambda x: df[x].count())
    df_cols['num_rows_missing'] = num_recs - df_cols.num_rows_values
    df_cols['pct_rows_missing'] = df_cols.num_rows_missing / num_recs
    df_cols['num_uniques'] = df_cols.cols.apply(lambda x: df[x].nunique())
    
    # get unique valeues for columns within unique value limits
    df_cats = df_cols[df_cols.num_uniques <= max_uniques]
    df_cats['unique_values'] = df_cats.cols.apply(lambda x: df[x].unique())
    
    # merge and set index
    join_type = 'inner' if limit_to_max else 'left'
    df_cols = df_cols.join(df_cats.unique_values, how=join_type)
    return df_cols.set_index('cols')


###############################################################################






###############################################################################
### functions stolen from instructors                                       ###
###############################################################################

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

# handle_missing_values(df, prop_required_column, prop_required_row).  
#   - The input: a dataframe, a number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column.  i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing), a number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row.  i.e. if prop_required_row = .75, then you are requiring a row to have at least 75% of variables with a non-missing value (no more that 25% missing) 
#   - The output: the dataframe with the columns and rows dropped as indicated. *Be sure to drop the columns prior to the rows in your function.*


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# Create a function that fills missing values with 0s where it makes sense based on the attribute/field/column/variable. 

def fill_zero(df, cols):
    df.fillna(value=0, inplace=True)
    return df


def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


# Data types: Write a function that takes in a dataframe and a list of columns names and returns the dataframe with the datatypes of those columns changed to a non-numeric type & use this function to appropriately transform any numeric columns that should not be treated as numbers
def numeric_to_category(df, cols):
    df[cols] = df[cols].astype('category')
    return df


# Write a function that accepts the zillow data frame and removes the outliers. 
def remove_outliers_iqr(df, columns, k=1.5):
    for col in columns:
        q75, q25 = np.percentile(df[col], [75,25])
        ob = k*stats.iqr(df[col])
        ub = ob + q75
        lb = q25 - ob
        df = df[df[col] <= ub]
        df = df[df[col] >= lb]
    return df


def get_how_far_outliers(s,k):
    '''
    get_how_far_outliers(s,k)
    RETURNS Series
    '''
    q1,q3=s.quantile([.25,.75])
    iqr=q3-q1
    bound = iqr*k
    upper_bound=q3+bound
    lower_bound=q1-bound
    
    return s.apply(lambda x:max([x-upper_bound,0]) if x > lower_bound else x-lower_bound)




###############################################################################
### functions stolen from Sam                                               ###
###############################################################################


def Sams_one_function(tbd=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style('whitegrid')
    
    # Show distributions of all numeric data
    for feature in telco.select_dtypes(include=[float, int]):
        sns.distplot(telco[feature])
        plt.show()
    
    # and for categorical...

    categorical = ['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
                            'multiple_lines', 'online_security', 'online_backup', 'device_protection',
                        'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing',
                        'churn', 'contract_type', 'internet_service_type', 'payment_type']

    for feature in categorical:
        sns.countplot(x=feature, palette="ch:.25", data=telco)
        plt.show()


def df_value_counts(df):
    for col in df.columns:
        print(f'{col}:')
        if df[col].dtype == 'object':
            col_count = df[col].value_counts()
        else:
            if df[col].nunique() >= 35:
                col_count = df[col].value_counts(bins=10, sort=False)
            else:
                col_count = df[col].value_counts()
        print(col_count)


def Sams_other_function(tbd=True):
    import matplotlib.patches as mpatches
    sns.scatterplot(x='waiting', y='predicted', data=faithful, alpha=0.8)
    sns.scatterplot(x='waiting', y='eruptions', data=faithful) 
    plt.title('Waiting Time Between Geyser Eruptions at Old Faithful')
    plt.yticks(np.arange(0,6))
    rmse = regression_errors(faithful.eruptions, faithful.predicted)[4]
    blue_patch = mpatches.Patch(color='blue', label='Predicted')
    orange_patch = mpatches.Patch(color='orange', label='Actual')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.annotate(s=f'RMSE: {rmse}', xy=(72,1))
    plt.show()



def split_store_data(df, train_prop=.66): 
    train_size = int(len(df) * train_prop)
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test