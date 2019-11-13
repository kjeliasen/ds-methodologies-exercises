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


