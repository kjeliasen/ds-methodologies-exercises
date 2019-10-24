# Exercises
# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us 
# over their lifetime. I have monthly charges and tenure, so I think I will be 
# able to use those two attributes as features to estimate total_charges. I 
# need to do this within an average of $5.00 per customer.

# Create a file, explore.py, that contains the following functions for 
# exploring your variables (features & target).




import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale

def plot_variable_pairs(dataframe):
    '''
    plot_variable_pairs(dataframe)
    NO RETURN

    From Exercises:
    Write a function, plot_variable_pairs(dataframe) that plots all of the 
    pairwise relationships along with the regression line for each pair.
    '''
    g = sns.PairGrid(dataframe)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)


def months_to_years(tenure_months, df):
    '''
    months_to_years(tenure_months, df)
    RETURNS dataframe

    From Exercises:
    Write a function, months_to_years(tenure_months, df) that returns your 
    dataframe with a new feature tenure_years, in complete years as a customer.
    '''
    newdf = pd.DataFrame(df)
    ty = tenure_months.apply(lambda x: int(x / 12))
    newdf['tenure_years'] = ty
    return newdf


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    '''
    plot_categorical_and_continuous_vars(categorical_var, continuous_var, df)
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


def get_uniques(df, max_uniques=10, target_col=''):
    '''
    get_uniques(df, max_uniques=10, target_col='')
    RETURNS summary dataframe

    Receives dataframe as input, examines all columns defined as objects, and
    returns a summary report with column name as its index.

    Report showss on the number of unique values (column 'nunuiques') within 
    each column and provides the unique values column('uniques') if the unique 
    value count is is less than or equal to max_uniques.

    If the input dataframe contains the target column, enter that name as the 
    target_col argument so it can be removed from the analysis.
    '''
    df_objs = pd.DataFrame(get_objs(df), columns=['cols'])
    df_objs = df_objs[df_objs.cols != target_col]
    df_objs['nuniques'] = df_objs.cols.apply(lambda x: df[x].nunique())
    df_cats = df_objs[df_objs.nuniques <= max_uniques]
    df_cats['uniques'] = df_cats.cols.apply(lambda x: df[x].unique())
    df_objs = df_objs.join(df_cats.uniques, how='left')
    return df_objs.set_index('cols')


def plot_violin(features, target, df, palette=['blue','orange']):
    '''
    plot_violin(features, target, df, palette=['blue','orange'])
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


def loopy_graphs(df, target):
    '''
    plot_violin(features, target, df, palette=['blue','orange'])
    NO RETURN

    Jeff's 'loopy_graphs' function
    '''
    features = list(df.columns[(df.dtypes == object) | (df.nunique()<5)])
    
    pop_rate = df[target].mean()
    for i, feature in enumerate(features):
        sns.barplot(feature,target,data=df,alpha=.6)
        plt.show()

