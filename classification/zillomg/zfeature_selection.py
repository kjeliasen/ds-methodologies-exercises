###############################################################################
### python imports                                                          ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LassoCV

import statsmodels.api as sm
from statsmodels.formula.api import ols

###############################################################################
### local imports                                                           ###
###############################################################################

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain



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

# print('Got Feature Selection')