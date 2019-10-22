###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_profiling as pdspro


###############################################################################
### local imports                                                           ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain


###############################################################################
### get db url                                                              ###
###############################################################################

@timeifdebug  # <--- DO NOT RUN ARGS DEBUG HERE! Will pass password info.
def get_db_url(user=user, password=password, host=host, database='employees'):
    '''
    get_db_url(user=user, password=password, host=host, database='zillow')
    RETURNS login url for selected mysql database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'


###############################################################################
### classification functions                                                ###
###############################################################################

@timeifdebug
def paste_df(splain=local_settings['SPLAIN']):
    return df_df(pd.read_clipboard(), splain=splain)


@timeifdebug
def excel_df(excel_path, splain=local_settings['SPLAIN']):
    return df_df(pd.read_excel(excel_path), splain=splain)


@timeifdebug
def google_df(sheet_url, splain=local_settings['SPLAIN']):
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return df_df(pd.read_csv(csv_export_url), splain=splain)


@timeifdebug
def sql_df(sql, db, splain=local_settings['SPLAIN']):
    db_url = get_db_url(database=db)
    return df_df(pd.read_sql(sql, db_url), splain=splain)

@timeifdebug
def df_df(dataframe, splain=local_settings['SPLAIN']):
    frame_splain(dataframe, splain=splain)
    return dataframe


@timeifdebug
def get_titanic_data(splain=local_settings['SPLAIN']):
    return sql_df(sql='SELECT * FROM passengers',db='titanic_db', splain=splain)

###############################################################################
### regression functions                                                    ###
###############################################################################

@timeifdebug
def wrangle_telco():
    get_database = 'telco_churn'
    telco_url = get_db_url(user=user, password=password, host=host, database=get_database)
    two_year_custs_sql = '''
    SELECT
        customer_id,
        monthly_charges,
        tenure,
        total_charges
    FROM
        customers
    WHERE
        contract_type_id = 3
    '''

    custs = pd.read_sql(two_year_custs_sql, telco_url)
    custs.total_charges.replace(r'^\s*$ ', np.nan, regex=True, inplace=True) 
    custs = custs[custs.total_charges != ' ']    
    custs = custs[custs.total_charges != '']    
    custs=custs.astype({'total_charges': float})
    custs = custs.set_index('customer_id')
    return custs    


