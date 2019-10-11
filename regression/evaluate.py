# Exercises

# prepare environment
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


###############################################################################
# 1. Load the tips dataset from either pydataset or seaborn.



###############################################################################
# 2. Fit a linear regression model (ordinary least squares) and compute yhat, 
# predictions of tip using total_bill. You may follow these steps to do that:



###############################################################################
# 3. import the method from statsmodels: 
# `from statsmodels.formula.api import ols`
# 
#     * fit the model to your data, where x = total_bill and y = tip: 
#     regr = ols('y ~ x', data=df).fit()
# 
#     * compute yhat, the predictions of tip using total_bill: 
#     df['yhat'] = regr.predict(df.x)
# 
#     * Create a file evaluate.py that contains the following functions.



###############################################################################
# 4. Write a function, plot_residuals(x, y, dataframe) that takes the feature, 
# the target, and the dataframe as input and returns a residual plot. 
# (hint: seaborn has an easy way to do this!)



###############################################################################
# 5. Write a function, regression_errors(y, yhat), that takes in y and yhat, 
# returns the sum of squared errors (SSE), explained sum of squares (ESS), 
# total sum of squares (TSS), mean squared error (MSE) and root mean squared 
# error (RMSE).



###############################################################################
# 6. Write a function, baseline_mean_errors(y), that takes in your target, y, 
# computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and 
# returns the error values (SSE, MSE, and RMSE).



###############################################################################
# 7. Write a function, better_than_baseline(SSE), that returns true if your 
# model performs better than the baseline, otherwise false.



###############################################################################
# 8. Write a function, model_significance(ols_model), that takes the ols model 
# as input and returns the amount of variance explained in your model, and the 
# value telling you whether the correlation between the model and the tip 
# value are statistically significant.


