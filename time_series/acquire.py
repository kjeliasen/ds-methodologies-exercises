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
def paste_df(splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    return check_df(pd.read_clipboard(), splain=splain, **kwargs)


@timeifdebug
def excel_df(excel_path, splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    return check_df(pd.read_excel(excel_path), splain=splain, **kwargs)


@timeifdebug
def google_df(sheet_url, splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return check_df(pd.read_csv(csv_export_url), splain=splain, **kwargs)


@timeifdebug
def csv_df(csv, splain=local_settings.splain, **kwargs):
    '''
    csv_df(csv, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    Reads a csv file into a dataframe, then sends the dataframe to check_df().
    '''
    csv = pd.read_csv(csv, **kwargs)
    return check_df(csv, splain=splain, **kwargs)


@timeifdebug
def sql_df(sql, db, splain=local_settings.splain, **kwargs):
    '''
    sql_df(sql, db, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    Reads a csv file into a dataframe, then sends the dataframe to check_df().
    '''
    db_url = get_db_url(database=db)
    return check_df(pd.read_sql(sql, db_url), splain=splain)

@timeifdebug
def check_df(dataframe, splain=local_settings.splain, **kwargs):
    '''
    check_df(dataframe, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    This function receives any dataframe, replaces null values with np.nan 
    and passes it through frame_splain(). If splain is true, frame_splain()
    will produce a report on the dataframe.
    '''
    dataframe.fillna(value=np.nan, inplace=True)
    frame_splain(dataframe, splain=splain, **kwargs)
    return dataframe




###############################################################################
### regression functions                                                    ###
###############################################################################

