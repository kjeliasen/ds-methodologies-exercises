###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

###############################################################################
### local imports                                                           ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain


###############################################################################
### get db url                                                              ###
###############################################################################

@timeifdebug  # <--- DO NOT RUN ARGS DEBUG HERE! Will pass password info.
def get_db_url(user, password, host, database):
    '''
    get_db_url(user=user, password=password, host=host, database='zillow')
    RETURNS login url for selected mysql database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'


###############################################################################
### classification functions                                                ###
###############################################################################




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


