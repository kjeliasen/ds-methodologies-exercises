import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import math

import wrangle
import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


# As a customer analyst, I want to know who has spent the most money with us 
# over their lifetime. I have monthly charges and tenure, so I think I will be 
# able to use those two attributes as features to estimate total_charges. I 
# need to do this within an average of $5.00 per customer.

# Create split_scale.py that will contain the functions that follow. Each 
# scaler function should create the object, fit and transform both train and 
# test. They should return the scaler, train df scaled, test df scaled. Be 
# sure your indices represent the original indices from train/test, as those 
# represent the indices from the original dataframe. Be sure to set a random 
# state where applicable for reproducibility!

# split_my_data(X, y, train_pct)
# standard_scaler()
# scale_inverse()
# uniform_scaler()
# gaussian_scaler()
# min_max_scaler()
# iqr_robust_scaler()


# acquire data and remove null values 
df = wrangle.wrangle_grades()

# verify acquisition
df.info()

### Test Train Split
train, test = train_test_split(df, train_size = .80, random_state = 123)


### Transform Data
train_scaled_data = scaler.transform(train)
test_scaled_data = scaler.transform(test)

# create object & fit
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
# transform train
train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# transform test
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


# If we wanted to return to original values:
train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train.index.values])
test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test.index.values])


### Uniform scaler
scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)

train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


### Gaussian (Normal)
# create scaler object using yeo-johnson method and fit to train
scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)

# apply to train
train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

# apply to test
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])


### MinMax
scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)

train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

### Robust Scaler

scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)

train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])



def split_my_data(df, target_column, train_pct=.75, random_state=None):
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=random_state)
    return X_train, X_test, y_train, y_test
    
def standard_scaler():
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(train) # fit the object

def scale_inverse():


def uniform_scaler():


def gaussian_scaler():


def min_max_scaler():


def iqr_robust_scaler():
