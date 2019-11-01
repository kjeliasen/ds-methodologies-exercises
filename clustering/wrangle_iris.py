###############################################################################
### Iris Data                                                               ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import sql_df, csv_df
from prep import encode_col

@timeifdebug
def get_iris_data(
    type='sql', 
    sql='', 
    db='iris_db', 
    csv='iris.csv', 
    splain=local_settings.splain, 
    **kwargs
    ):
    '''
    get_iris_data(
        type='sql', 
        sql='', 
        db='iris_db', 
        csv='iris.csv', 
        splain=local_settings.splain, 
        **kwargs
    )
    RETURNS: dataframe

    Iris data is available as sql or csv, so this function allows for both 
    options. Default is sql and there is a default sql script in the function,
    but this may be overrided by the user.

    The output of this function passes through sql_df() and check_df().
    '''
    if type == 'csv':
        return csv_df(csv, splain=splain)
    if type == 'sql':
        set_sql = '''
    SELECT 
        m.measurement_id,
        m.sepal_length,
        m.sepal_width,
        m.petal_length,
        m.petal_width,
        s.species_name
    FROM 
        measurements m
    JOIN 
        species s
        USING(species_id)
        '''
        use_sql = set_sql if sql == '' else sql
        return sql_df(sql=use_sql, db=db, splain=splain, **kwargs)


@timeifdebug
def prep_iris_data(splain=local_settings.splain, **kwargs):
    '''
    prep_iris(splain=local_settings.splain, **kwargs)
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


