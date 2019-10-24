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

@timeifdebug
def encode_col(df, col):
    '''
    encode_col(df, col)
    RETURNS: df, encoder
    '''
    encoder = LabelEncoder()
    encoder.fit (df[[col]])
    df[[col]] = encoder.transform(df[[col]])
    return df, encoder


@timeifdebug
def simpute(df, column, missing_values=np.nan, strategy='most_frequent', splain=local_settings.splain):
    '''
    simpute(df, column, missing_values=np.nan, strategy='most_frequent', splain=local_settings.splain)
    RETURNS: df
    '''
    df[[column]] = df[[column]].fillna(missing_values)
    imp_mode = SimpleImputer(missing_values=missing_values, strategy=strategy)
    df[[column]] = imp_mode.fit_transform(df[[column]])
    return df


###############################################################################
###############################################################################
###############################################################################
# Iris Data

@timeifdebug
def prep_iris(splain=local_settings.splain):
    '''
    prep_iris(splain=local_settings.splain)
    RETURNS: df, encoder

    Iris Data

    1. Use the function defined in acquire.py to load the iris data.
    2. Drop the species_id and measurement_id columns.
    3. Rename the species_name column to just species.
    4. Encode the species name using a sklearn label encoder. Research the 
    inverse_transform method of the label encoder. How might this be useful?
    5. Create a function named prep_iris that accepts the untransformed iris 
    data, and returns the data with the transformations above applied.
    '''
    df = get_iris_data(type='sql', splain=splain)
    df = df.drop(columns='measurement_id', axis=1)
    df = df.rename(columns={'species_name': 'species'})
    df, encoder = encode_col(df=df, col='species')
    return df, encoder


###############################################################################
###############################################################################
###############################################################################
# Titanic Data

@timeifdebug
def prep_titanic(splain=local_settings.splain):
    '''
    prep_titanic(splain=local_settings.splain)
    RETURNS: df, encoder, scaled
    
    
    # Titanic Data

    # 1. Use the function you defined in acquire.py to load the titanic data set.
    # 2. Handle the missing values in the embark_town and embarked columns.
    # 3. Remove the deck column.
    # 4. Use a label encoder to transform the embarked column.
    # 5. Scale the age and fare columns using a min max scaler. Why might this be 
    # beneficial? When might you not want to do this?
    # 6. Create a function named prep_titanic that accepts the untransformed 
    # titanic data, and returns the data with the transformations above applied.

    # Note: drop columns updated to deck, embarked, passenger_id in explore
    # Note: encoding changed to embark_town
    '''
    df = get_titanic_data(splain=splain)
    df.drop(columns=['deck', 'embarked','passenger_id'], inplace=True)
    df = simpute(df=df, column='embark_town', splain=splain)
    df, encoder = encode_col(df=df, col='embark_town')
    scaled = MinMaxScaler()
    scaled.fit(df[['age','fare']])
    df[['age','fare']] = scaled.transform(df[['age','fare']])
    return df, encoder, scaled


