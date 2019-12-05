###############################################################################
### python imports                                                          ###
###############################################################################

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
from sklearn import metrics


###############################################################################
### local imports                                                           ###
###############################################################################

import acquire as acq
import prepare as prep
import explore as xplr
import evaluate as meval

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from dfo import DFO


###############################################################################
# Fit the logistic regression classifier to your training sample and transform, 
# i.e. make predictions on the training sample



###############################################################################
# Evaluate your in-sample results using the model score, confusion matrix, and 
# classification report.



###############################################################################
# Print and clearly label the following: 
# Accuracy, 
# true positive rate, 
# false positive rate, 
# true negative rate, 
# false negative rate, 
# precision, 
# recall, 
# f1-score, and 
# support.



###############################################################################
# Look in the scikit-learn documentation to research the solver parameter. What 
# is your best option(s) for the particular problem you are trying to solve and 
# the data to be used?



###############################################################################
# Run through steps 2-4 using another solver (from question 5)



###############################################################################
# Which performs better on your in-sample data?

# Shows RMSE for baseline model using mean of calories burned

def baseline_rmse(train):
    train['avg_cals_burned'] = train.cals_burned.mean()
    train['last_week_cals'] = train.cals_burned.shift(7)
    score_train = train[train.last_week_cals >0]
    score_train.head()
    rms_mean = sqrt(metrics.mean_squared_error(score_train.cals_burned,score_train.avg_cals_burned))
    print("The RMSE using the median calories burned: ", rms_mean)

    # RMSE for a predictive model using the prior week calories burned + the delta

def one_week_rmse(train):
    score_train = train[train.last_week_cals >0]
    rms_prior_week = sqrt(metrics.mean_squared_error(score_train.cals_burned,score_train.last_week_cals))
    print("The RMSE using a one week shift for calories burned: ", round(rms_prior_week,5))

def active_rmse(train):
    train['avg_cals_burned'] = train.cals_burned.mean()
    train['last_week_cals'] = train.cals_burned.shift(7)
    score_train = train[train.last_week_cals >0]
    score_train.head()
    rms_mean = sqrt(metrics.mean_squared_error(score_train.cals_burned,score_train.avg_cals_burned))
    print("The RMSE using the median calories post July 7th: ", rms_mean)

def inactive_rmse(train):
    train['avg_cals_burned'] = train.cals_burned.mean()
    train['last_week_cals'] = train.cals_burned.shift(7)
    score_train = train[train.last_week_cals >0]
    score_train.head()
    rms_mean = sqrt(metrics.mean_squared_error(score_train.cals_burned,score_train.avg_cals_burned))
    print("The RMSE using the median calories pre July 7th: ", rms_mean)