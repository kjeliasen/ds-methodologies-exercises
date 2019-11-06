###############################################################################
### python imports                                                          ###
###############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

import acquire as acq
import prepare as prep

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import get_iris_data, get_titanic_data
from dfo import DFO



###############################################################################
def plot_residuals(x, y, dataframe):
    '''
    plot_residuals(x, y, dataframe)

    4. Write a function, plot_residuals(x, y, dataframe) that takes the feature, 
    the target, and the dataframe as input and returns a residual plot. 
    (hint: seaborn has an easy way to do this!)
    '''
    return sns.residplot(x=x, y=y, data=dataframe)


###############################################################################
def regression_errors(y, yhat):
    '''
    regression_errors(y, yhat)
    returns SSE, ESS, TSS, MSE, RMSE

    5. Write a function, regression_errors(y, yhat), that takes in y and yhat, 
    returns the sum of squared errors (SSE), explained sum of squares (ESS), 
    total sum of squares (TSS), mean squared error (MSE) and root mean squared 
    error (RMSE).
    '''
    ESS = sum((yhat - y.mean())**2)
    MSE = mean_squared_error(y, yhat)
    SSE = MSE*len(y)
    TSS = ESS + SSE
    RMSE = sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE

###############################################################################
def make_baseline(df):
    '''
    6. Write a function, baseline_mean_errors(y), that takes in your target, y, 
    computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and 
    returns the error values (SSE, MSE, and RMSE).
    '''
    # copy dataframe
    df_baseline = df[['x', 'y']]
    # compute the overall mean of the y values and add to 'yhat' as our prediction
    df_baseline['yhat'] = df_baseline['y'].mean()
    # compute the difference between y and yhat
    df_baseline['residual'] = df_baseline['yhat'] - df_baseline['y']
    # square that delta
    df_baseline['residual^2'] = df_baseline['residual'] ** 2
    return df_baseline

def baseline_mean_errors(df_baseline):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(df_baseline.y, df_baseline.yhat)
    return SSE, ESS, TSS, MSE, RMSE


###############################################################################
def better_than_baseline(sse, blsse):
    '''
    7. Write a function, better_than_baseline(SSE), that returns true if your 
    model performs better than the baseline, otherwise false.
    '''
    return sse < blsse

def is_isnot(is_it):
    return 'is' if is_it else 'is not'


def model_significance(ols_model):
    '''
    8. Write a function, model_significance(ols_model), that takes the ols model 
    as input and returns the amount of variance explained in your model, and the 
    value telling you whether the correlation between the model and the tip 
    value are statistically significant.
    '''
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return r2, f_pval

