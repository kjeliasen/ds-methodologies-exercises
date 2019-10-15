# Exercises
# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us 
# over their lifetime. I have monthly charges and tenure, so I think I will be 
# able to use those two attributes as features to estimate total_charges. I 
# need to do this within an average of $5.00 per customer.

# prepare enviornment

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

# For Problem 1:
from sklearn.feature_selection import SelectKBest, f_regression 
# For Problem 3:
import statsmodels.api as sm

###############################################################################
# 
# 1. Write a function, `select_kbest_freg_unscaled()` that takes X_train, y_train 
# and k as input (X_train and y_train should not be scaled!) and returns a 
# list of the top k features.

def select_kbest_freg_unscaled(X_train, y_train, k=2):
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_train, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature, f_selector


###############################################################################
# 
# 2. Write a function, `select_kbest_freg_scaled()` that takes X_train, y_train 
# (scaled) and k as input and returns a list of the top k features.

def select_kbest_freg_scaled(X_train_scaled, y_train_scaled, k=2):
    f_feature_scaled, f_selector_scaled = select_kbest_freg_unscaled(X_train=X_train_scaled, y_train=y_train_scaled, k=k)
    return f_feature_scaled, f_selector_scaled


###############################################################################
# 
# 3. Write a function, `ols_backward_elimination()` that takes X_train and 
# y_train (scaled) as input and returns selected features based on the ols 
# backwards elimination method.

def ols_backward_elimination(X_train_scaled, y_train_scaled):
    cols = list(X_train_scaled.columns)
    cols_removed = []
    removed_pvals = []
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X_train[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y_train_scaled,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols_removed.append(feature_with_p_max)
            removed_pvals.append(pmax)
            cols.remove(feature_with_p_max)
        else:
            break
            
    selected_features_BE = cols
    removed_features_BE = cols_removed
    removed_pvals_BE = removed_pvals
    return selected_features_BE, removed_features_BE, removed_pvals_BE


###############################################################################
# 
# 4. Write a function, `lasso_cv_coef()` that takes X_train and y_train as 
# input and returns the coefficients for each feature, along with a plot of 
# the features and their weights.


###############################################################################
# 
# 5. Write 3 functions: 
# 
###############################################################################
# 
# The first computes the number of optimum features (n) using rfe
# 



###############################################################################
# 
# The second takes n as input and returns the top n features
# 



###############################################################################
# 
# The third takes the list of the top n features as input and returns a new 
# X_train and X_test dataframe with those top features , 
# `recursive_feature_elimination()` that computes the optimum number of 
# features (n) and returns the top n features.


