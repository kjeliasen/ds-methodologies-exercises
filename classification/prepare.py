
# Exercises
# The end product of this exercise should be the specified functions in a 
# python script named prepare.py. Do these in your 
# classification_exercises.ipynb first, then transfer to the prepare.py file.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


import acquire as acq
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain

from acquire import get_iris_data, get_titanic_data


def encode_col(df, col):
    encoder = LabelEncoder()
    encoder.fit (df[col])
    df[col] = encoder.transform(df[col])
    return encoder


# Iris Data

# Use the function defined in acquire.py to load the iris data.
# Drop the species_id and measurement_id columns.
# Rename the species_name column to just species.
# Encode the species name using a sklearn label encoder. Research the 
# inverse_transform method of the label encoder. How might this be useful?
# Create a function named prep_iris that accepts the untransformed iris data, 
# and returns the data with the transformations above applied.

def prep_iris(splain=local_settings.splain):
    df = get_iris_data(type='sql', splain=splain)
    df = df.drop(columns='measurement_id', axis=1)
    df = df.rename(columns={'species_name': 'species'})
    encoder = encode_col(df=df, col='species')
    return df, encoder


# Titanic Data

# Use the function you defined in acquire.py to load the titanic data set.
# Handle the missing values in the embark_town and embarked columns.
# Remove the deck column.
# Use a label encoder to transform the embarked column.
# Scale the age and fare columns using a min max scaler. Why might this be beneficial? When might you not want to do this?
# Create a function named prep_titanic that accepts the untransformed titanic data, and returns the data with the transformations above applied.

def prep_titanic(splain=local_settings.splain):
    df = get_titanic_data(splain=splain)
    df = df.drop(columns=['deck'])
    df = simpute(df=df, column='embarked')
    df = simpute(df=df, column='embark_town')
    encoder = encode_col(df=df, col='embarked')
    scaled = MinMaxScaler()
    scaled.fit(df[['age','fare']])
    df[['age','fare']] = scaled.transform(df[['age','fare']])
    return df, encoder, scaled



def simpute(df, column, missing_values=np.nan, strategy='most_frequent'):
    imp_mode = SimpleImputer(missing_values=missing_values, strategy=strategy)
    #imp_mode.fit(df[[column]])
    #df[[column]] = imp_mode.transform(df[[column]])
    df[[column]] = imp_mode.fit_transform(df[[column]])
    #return df

