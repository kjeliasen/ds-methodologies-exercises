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
# For Problem 4:
from sklearn.linear_model import LassoCV
import matplotlib
# For Problem 5:
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# For Problem 3:
def get_fit_summary(y_train_scaled, X_train_scaled):
    '''
get_fit_summary(y_train_scaled, X_train_scaled)
RETURNS: fit.summary()
    '''
    # create the OLS object:
    ols_model = sm.OLS(y_train_scaled, X_train_scaled)
    # fit the model:
    fit = ols_model.fit()
    # summarize:
    return fit.summary()


###############################################################################
# 
# 1. Write a function, `select_kbest_freg_unscaled()` that takes X_train, y_train 
# and k as input (X_train and y_train should not be scaled!) and returns a 
# list of the top k features.

def select_kbest_freg_unscaled(X_train, y_train, k=2):
    '''
select_kbest_freg_unscaled(X_train, y_train, k=2)
RETURNS: f_feature, f_selector
    '''
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
    '''
select_kbest_freg_unscaled(X_train_scaled, y_train_scaled, k=2)
RETURNS: f_feature_scaled, f_selector_scaled
    '''
    f_feature_scaled, f_selector_scaled = select_kbest_freg_unscaled(X_train=X_train_scaled, y_train=y_train_scaled, k=k)
    return f_feature_scaled, f_selector_scaled


###############################################################################
# 
# 3. Write a function, `ols_backward_elimination()` that takes X_train and 
# y_train (scaled) as input and returns selected features based on the ols 
# backwards elimination method.

def ols_backward_elimination(X_train_scaled, y_train_scaled):
    '''
ols_backward_elimination(X_train_scaled, y_train_scaled)
RETURNS selected_features_BE, removed_features_BE, removed_pvals_BE
    '''
    cols = list(X_train_scaled.columns)
    cols_removed = []
    removed_pvals = []
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X_train_scaled[cols]
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
    return model, alpha, score, coef, yhat


###############################################################################
# 
# 5. Write 3 functions: 
# 
###############################################################################
# 
# The first computes the number of optimum features (n) using rfe
# 

def optimal_number_of_features(X_train, y_train, X_test, y_test):
    '''
    optimal_number_of_features(X_train, y_train, X_test, y_test)
    RETURNS: number_of_features
    
    discover the optimal number of features, n, using our scaled x and y dataframes, recursive feature
    elimination and linear regression (to test the performance with each number of features).
    We will use the output of this function (the number of features) as input to the next function
    optimal_features, which will then run recursive feature elimination to find the n best features

    Shamelessly stolen from David Espinola
    '''

    number_of_attributes = X_train.shape[1]
    number_of_features_list=np.arange(1,number_of_attributes)
    high_score=0
    
    #Variable to store the optimum features
    number_of_features=0           
    score_list =[]
    
    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features


###############################################################################
# 
# The second takes n as input and returns the top n features
# 

def optimal_features(X_train, y_train, number_of_features):
    '''
    optimal_features(X_train, y_train, number_of_features)
    RETURNS: selected_features_rfe
    
    Taking the output of optimal_number_of_features, as n, and use that value to 
    run recursive feature elimination to find the n best features
    
    Shamelessly stolen from David Espinola
    '''

    cols = list(X_train.columns)
    model = LinearRegression()
    
    #Initializing RFE model
    rfe = RFE(model, number_of_features)

    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train,y_train)  

    #Fitting the data to model
    model.fit(X_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    return selected_features_rfe

###############################################################################
# 
# The third takes the list of the top n features as input and returns a new 
# X_train and X_test dataframe with those top features
# 

def create_optimal_dataframe(X_train, X_test, selected_features_rfe):
    '''
    create_optimal_dataframe(X_train, X_test, selected_features)
    RETURNS X_train_optimal, X_test_optimal

    Takes output of optimal_features and creates new optimized X_train and X_test
    dataframes containing only those features..
    '''

    X_train_optimal = X_train[selected_features_rfe]
    X_test_optimal = X_test[selected_features_rfe]

    return X_train_optimal, X_test_optimal



###############################################################################
#  
# `recursive_feature_elimination()` that computes the optimum number of 
# features (n) and returns the top n features.

def recursive_feature_elimination(X_train, y_train, X_test, y_test):
    '''
    recursive_feature_elimination(X_train, y_train, X_test, y_test)
    RETURNS X_train_optimal, X_test_optimal

    Combines optimal_number_of_features, optimal_features, and 
    create_optimal_dataframe into one single function. Accepts X and y train and 
    test dataframes, returns optimal X train and test dataframes.
    '''

    number_of_features = optimal_number_of_features(X_train, y_train, X_test, y_test)
    selected_features_rfe = optimal_features(X_train, y_train, number_of_features)
    X_train_optimal, X_test_optimal = create_optimal_dataframe(X_train, X_test, selected_features_rfe)

    return X_train_optimal, X_test_optimal