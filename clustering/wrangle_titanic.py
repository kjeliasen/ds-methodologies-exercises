###############################################################################
### Titanic Data                                                            ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import sql_df
from prepare import encode_col, simpute, MinMaxScaler

@timeifdebug
def get_titanic_data(splain=local_settings.splain, **kwargs):
    '''
    get_titanic_data(splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    The Titanic dataset required for future use comes from a pre-provided sql
    statement. This function passes through sql_df() and check_df().
    '''
    return sql_df(sql='SELECT * FROM passengers',db='titanic_db', splain=splain, **kwargs)


@timeifdebug
def prep_titanic_data(splain=local_settings.splain, **kwargs):
    '''
    prep_titanic(splain=local_settings.splain, **kwargs)
    RETURNS: df, encoder, scaler
    
    
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
    scaler = MinMaxScaler()
    scaler.fit(df[['age','fare']])
    df[['age','fare']] = scaler.transform(df[['age','fare']])
    return df, encoder, scaler



