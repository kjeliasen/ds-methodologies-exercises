###############################################################################
### python imports                                                          ###
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from math import sqrt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LassoCV

import warnings
warnings.filterwarnings("ignore")

# from math import sqrt
import statsmodels.api as sm
from statsmodels.formula.api import ols

###############################################################################
### local imports                                                           ###
###############################################################################

import acquire as acq
import prepare as prep

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
#from acquire import get_iris_data, get_titanic_data
from dfo import DFO


###############################################################################
### plotting functions                                                      ###
###############################################################################

def df_limit_numerics(df):
    new_df = df[list(df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64') | (df.dtypes == 'bool')])]
    return new_df


def all_line_plots(df):
    for col in df_limit_numerics(df).columns:
        plt.figure(figsize=(7,5))
        df[col].plot()
        plt.ylabel(col)
        plt.show()


def all_box_plots(df, x_col='week_day'):
    new_df = df.copy()
    new_df = new_df.sort_values(by=x_col)
    for col in df_limit_numerics(df).columns:
        
        plt.figure(figsize=(7,5))
        sns.boxplot(data=new_df,x=x_col,y=col,showmeans=True)

        plt.hlines(df[col].mean(),0,6, color='limegreen')
        plt
        plt.ylabel(col)
        plt.show()


@timeifdebug
def plot_variable_pairs(dataframe, **kwargs):
    '''
    plot_variable_pairs(dataframe, **kwargs)
    NO RETURN

    From Exercises:
    Write a function, plot_variable_pairs(dataframe) that plots all of the 
    pairwise relationships along with the regression line for each pair.
    '''
    g = sns.PairGrid(dataframe)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)


@timeifdebug
def months_to_years(tenure_months, df, **kwargs):
    '''
    months_to_years(tenure_months, df, **kwargs)
    RETURNS dataframe

    From Exercises:
    Write a function, months_to_years(tenure_months, df) that returns your 
    dataframe with a new feature tenure_years, in complete years as a customer.
    '''
    newdf = pd.DataFrame(df)
    ty = tenure_months.apply(lambda x: int(x / 12))
    newdf['tenure_years'] = ty
    return newdf


@timeifdebug
def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df, **kwargs):
    '''
    plot_categorical_and_continuous_vars(categorical_var, continuous_var, df, **kwargs)
    NO RETURN

    From Exercises:
    Write a function, plot_categorical_and_continous_vars(categorical_var, 
    continuous_var, df), that outputs 3 different plots for plotting a 
    categorical variable with a continuous variable, e.g. tenure_years with 
    total_charges. For ideas on effective ways to visualize categorical with 
    continuous: https://datavizcatalogue.com/. You can then look into seaborn 
    and matplotlib documentation for ways to create plots
    '''
    xvals=categorical_var
    yvals=continuous_var
    p1 = sns.barplot(x=xvals, y=yvals, data=df)
    plt.show()
    p2 = sns.boxplot(x=xvals, y=yvals, data=df)
    plt.show()
    p3 = sns.stripplot(x=xvals, y=yvals, data=df)
    plt.show()


def get_objs(df, **kwargs):
    '''
    get_objs(df, **kwargs)
    RETURNS pd Series
    '''
    return df.columns[df.dtypes=='object']


@timeifdebug
def get_uniques(
        df, 
        get_cols=[], 
        max_uniques=10, 
        target_col='', 
        **kwargs
        ):
    '''
    get_uniques(
        df, 
        get_cols=[], 
        max_uniques=10, 
        target_col='', 
        **kwargs
        )
    RETURNS summary dataframe

    Receives dataframe as input, examines all columns defined as objects, and
    returns a summary report with column name as its index.

    Report showss on the number of unique values (column 'nunuiques') within 
    each column and provides the unique values column('uniques') if the unique 
    value count is is less than or equal to max_uniques.

    If the input dataframe contains the target column, enter that name as the 
    target_col argument so it can be removed from the analysis.
    '''
    cols = [col for col in get_cols if col in df.columns]
    #df_objs = pd.DataFrame(get_objs(df), columns=['cols'])
    df_objs = pd.DataFrame(get_objs(df), columns=['cols'])
    df_objs = df_objs[df_objs.cols != target_col]
    df_objs['nuniques'] = df_objs.cols.apply(lambda x: df[x].nunique())
    df_cats = df_objs[df_objs.nuniques <= max_uniques]
    df_cats['uniques'] = df_cats.cols.apply(lambda x: df[x].unique())
    df_objs = df_objs.join(df_cats.uniques, how='left')
    return df_objs.set_index('cols')


@timeifdebug
def plot_violin(features, target, df, palette=['blue','orange'], **kwargs):
    '''
    plot_violin(features, target, df, palette=['blue','orange'], **kwargs)
    NO RETURN

    Dom's 'plot_violin' function
    '''
    for descrete in df[features].select_dtypes([object,int]).columns.tolist():
        if df[descrete].nunique() <= 5:
            for continous in df[features].select_dtypes(float).columns.tolist():
                sns.violinplot(descrete, continous, hue=target,
                data=df, split=True, palette=palette)
                plt.title(continous + 'x' + descrete)
                plt.ylabel(continous)
                plt.show()


@timeifdebug
def loopy_graphs(df, target, **kwargs):
    '''
    loopy_graphs(df, target, **kwargs)
    NO RETURN

    Jeff's 'loopy_graphs' function
    '''
    features = list(df.columns[(df.dtypes == object) | (df.nunique()<5)])
    
    pop_rate = df[target].mean()
    for i, feature in enumerate(features):
        sns.barplot(feature,target,data=df,alpha=.6)
        plt.show()


@timeifdebug
def get_fit_summary(model):
    # fit the model:
    fit = model.fit()
    return fit.summary()
    
    
@timeifdebug
def get_ols_model(X_train, y_train, train):
#    model = ols('y_train ~ X_train', data=train).fit()
    model = ols('y_train ~ X_train').fit()
    yhat = ols_model.predict(y_train)
    return model, yhat


@timeifdebug
def lasso_cv_coef(X_train, y_train, plotit=True, summarize=True):
    '''
    lasso_cv_coef(X_train, y_train, plotit=True, summarize=True)
    plotit produces plot at runtime
    summarize returns printed summary
    RETURNS: model, alpha, score, coef, yhat    
    '''
    model = LassoCV().fit(X_train, y_train)
    alpha = model.alpha_
    score = model.score(X_train, y_train)
    coef = pd.Series(model.coef_, index = X_train.columns)
    yhat = model.predict(X_train)
    if summarize:
        imp_coef = coef.sort_values()
        vars_kept = sum(coef != 0)
        vars_elim = sum(coef == 0)
        print("Best alpha using built-in LassoCV: %f" %model.alpha_)
        print("Best score using built-in LassoCV: %f" %model.score(X_train,y_train))
        print("Lasso picked " + str(sum(coef != 0)) + 
              " variables and eliminated the other " +  
              str(sum(coef == 0)) + " variables")
        print(pd.DataFrame(coef))
    if plotit:
        imp_coef = coef.sort_values()
        matplotlib.rcParams['figure.figsize'] = (4.0, 5.0)
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        plt.plot()
    return model, yhat, alpha, score, coef



@timeifdebug
def bool_ttest_single_tf(df, target_column='target', column_list=[], alpha=.05, **kwargs):

    p_list = []
    col_name = target_column
    for col in column_list:
        is_true = df[df[col] == True]
        y_if_true = is_true[col_name].mean()
        y_mean = df[col_name].mean()
        t, p = stats.ttest_1samp(is_true.churn, df.churn.mean())
        p_list.append([col, t, p, p<alpha, y_if_true, y_mean])
    df_p = pd.DataFrame(p_list, columns=['column','t_stat','p_value','is_sig', col_name + '_if_true', col_name + '_mean']).set_index('column').sort_values(by='p_value')
    return df_p


@timeifdebug
def bool_ttest_multi_tf(df, target_column='target', column_list=[], alpha=.05, **kwargs):
    col_name = target_column
    p_list = []
    for col in column_list:
        is_true = df[df[col] == True][col_name]
        is_not = df[df[col] != True][col_name]
        y_if_true = is_true.mean()
        y_if_not = is_not.mean()
        t, p = stats.ttest_ind(is_true, is_not)
        p_list.append([col, t, p, p<alpha, y_if_true, y_if_not])

    df_p2 = pd.DataFrame(p_list, columns=['column','t_stat','p_value','is_sig', col_name + '_if_true', col_name + '_if_not']).set_index('column').sort_values(by='p_value')
    return df_p2


@timeifdebug
def bool_ttest_single_uniq(df, target_column='target', check_column='check', alpha=.05, **kwargs):

    p_list = []
    uniq_vals = sorted(df[check_column].unique())
    print(uniq_vals)
    for val in uniq_vals:
        is_true = df[df[check_column] == val]
        num_true = len(is_true)
        y_if_true = is_true[target_column].mean()
        std_if_true = is_true[target_column].std()
        y_mean = df[target_column].mean()
        y_std = df[target_column].std()
        t, p = stats.ttest_1samp(is_true[target_column], df[target_column].mean())
        p_list.append([check_column, val, num_true, t, p, p<alpha, y_if_true, y_mean, std_if_true, y_std])
    df_p = (pd.DataFrame(
        p_list, 
        columns=[
            'column','value','count','t_stat','p_value','is_sig',
            target_column+'_if_value',target_column+'_mean',
            'std_if_value','std_all'])
        .set_index(['value'])
            .sort_values(by='p_value'))
    return df_p

# @timeifdebug
# def bool_ttest_multi_uniq(df, target_column='target', column_list=[], compare_val=True, alpha=.05, **kwargs):
#     col_name = target_column
#     p_list = []
#     for col in column_list:
#         is_true = df[df[col] == compare_val][col_name]
#         is_not = df[df[col] != compare_val][col_name]
#         y_if_true = is_true.mean()
#         y_if_not = is_not.mean()
#         t, p = stats.ttest_ind(is_true, is_not)
#         p_list.append([col, t, p, p<alpha, y_if_true, y_if_not])

#     df_p2 = pd.DataFrame(p_list, columns=['column','t_stat','p_value','is_sig', col_name + '_if_true', col_name + '_if_not']).set_index('column').sort_values(by='p_value')
#     return df_p2




if __name__ == '__main__':
    print('running from main')


def activity_stack(df):
    # Make data
    cols = ['mins_heavy','mins_mod','mins_light','mins_idle']
    pal = ['#a8122c','#e5c100','#668466','#ffffff']
    data = df[cols].resample('W').mean()

    # We need to transform the data from raw data to percentage (fraction)
    data_perc = data.divide(data.sum(axis=1), axis=0)

    # Make the plot
    plt.stackplot(data.index, data_perc['mins_heavy'], data_perc['mins_mod'], data_perc['mins_light'], data_perc['mins_idle'], labels=cols,colors=pal)
    # plt.stackplot(data)
    plt.legend(loc='upper left')
    plt.margins(0,0)
    axes=plt.gca()
    axes.set_ylim([0,0.25])
    plt.title('Time spent by activity intensity (first 25% shown)')
    plt.show()


def pre_july(df):
    title = "Pre 7/7 Averages"
    df_early = df[df.index < '2018-07-07']
    df_early = df_early[['cals_burned','dist','mins_heavy']]
    print('\033[1m' + title + '\033[0m')
    return df_early.describe().loc[['mean']]

def post_july(df):
    title = "Post 7/7 Averages"
    df_late = df[df.index > '2018-07-07']
    df_late = df_late[['cals_burned','dist','mins_heavy']]
    print('\033[1m' + title + '\033[0m')
    return df_late.describe().loc[['mean']]

#boxplots by pre and post July 7th

def all_box_plots_pre_post(df, x_col='pre_post'):
    df['pre_post'] = np.where(df.index > '2018-07-07',"post",'pre')
    new_df = df.copy()
    new_df = new_df.sort_values(by=x_col,ascending=False)
    for col in df_limit_numerics(df).columns:
        
        plt.figure(figsize=(7,3))
        sns.boxplot(data=new_df,x=x_col,y=col,showmeans=True)

        plt.hlines(df[col].mean(),0,2, color='limegreen')
        plt
        plt.ylabel(col)
        plt.show()