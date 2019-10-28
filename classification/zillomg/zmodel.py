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

import statsmodels.api as sm
from statsmodels.formula.api import ols

from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain


