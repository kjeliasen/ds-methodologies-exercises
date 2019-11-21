###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import json

from os import path


###############################################################################
### local imports                                                           ###
###############################################################################

#from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain



###############################################################################
### get db url                                                              ###
###############################################################################

# @timeifdebug  # <--- DO NOT RUN ARGS DEBUG HERE! Will pass password info.
# def get_db_url(user=user, password=password, host=host, database='employees'):
#     '''
#     get_db_url(user=user, password=password, host=host, database='zillow')
#     RETURNS login url for selected mysql database
#     '''
#     return f'mysql+pymysql://{user}:{password}@{host}/{database}'


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
    frame_splain(dataframe, splain=splain, debug=True, print_disp='display', **kwargs)
    return dataframe




###############################################################################
### web-scraping functions                                                  ###
###############################################################################

def get_val_from_key(dict={}, keys=None, key='key'):
    if keys is None:
        keys = dict.keys()
    return dict[key] if key in keys else None


@timeifdebug
def get_json_payload_data(
    target_table='items',
    base_url='https://python.zach.lol',
    sect_key='items',
    api_url='/api/v1/items',
    idx_col='item_id',
    beg_page=1,
    csv_name='items.csv',
    data_key='payload',
    status_key='status',
    on_page_key='page',
    of_pages_key='max_page',
    url_next_key='next_page',
    url_prev_key='previous_page',
    url_get_page='?page=',
    sep=',',
    show_log=False, 
    use_cache=True,
    to_csv=False,
    debug=False,
    splain=True,
    maxcols=15,
    title='table'):
    
    # check for immediate exit
    if use_cache:
        if path.exists(csv_name):
            df = check_df(pd.read_csv(csv_name), splain=splain, title=title, maxcols=maxcols)
            return df 
        to_csv = True

    # Setup df
    tdf = pd.DataFrame()

    # Initialize variables
    cur_page = beg_page
    pages_max = -1
    keep_going = True
    
    # Get initial page
    next_url = api_url
    while keep_going:
        is_complete = False
        get_url = base_url + next_url
        
        # Get webpage
        if show_log:
            print('Fetching page', get_url)
        response = requests.get(get_url)
        
        # Get JSON data
        response_json = response.json()
        json_keys = response_json.keys()
        
        # Get payload
        payload = get_val_from_key(dict=response_json, keys=json_keys, key=data_key)
        if payload is None:
            break

        payload_keys = payload.keys()
        
        # Set navigation values
        on_page = get_val_from_key(dict=payload, keys=payload_keys, key=on_page_key)
        of_pages = get_val_from_key(dict=payload, keys=payload_keys, key=of_pages_key)
        url_next = get_val_from_key(dict=payload, keys=payload_keys, key=url_next_key)
        url_prev = get_val_from_key(dict=payload, keys=payload_keys, key=url_prev_key)
        
        # Get target data
        target_data = get_val_from_key(dict=payload, keys=payload_keys, key=sect_key)
        if target_data is None:
            break
        
        # Make page dataframe
        page_df = pd.DataFrame(target_data)
        if idx_col in page_df.columns:
            page_df.set_index(idx_col, inplace=True)
        else:
            print('index is missing')
            break
        
        tdf = tdf.append(page_df, verify_integrity=True)

        if url_next is None:
            keep_going = False
            
        if keep_going:
            next_url = url_next
        else:
            next_url = None
        
        is_complete = True
    
    df = check_df(tdf, splain=splain, title=title, maxcols=maxcols)

    if to_csv:
        df.to_csv(
            path_or_buf=csv_name, 
            sep=sep, 
            index=True, 
            index_label=idx_col, 
        )
        
    return df
    

@timeifdebug
def output_payload_data(
    target_table='items', 
    base_url='https://python.zach.lol',
    show_log=False,
    sect_url_keys={},
    sep=',',
    debug=False,
    splain=True, 
    use_cache=True,
    to_csv=False,
    maxcols=15,
    **kwargs):
    
    url_keys = sect_url_keys[target_table]
    sect_key = url_keys['sect_key']
    api_url = url_keys['api_url']
    idx_col = url_keys['idx_col']
    beg_page=url_keys['page_beg']
    csv_name=url_keys['csv_name']
    
    df = get_json_payload_data(
        target_table=target_table,
        base_url=base_url,
        sect_key=sect_key,
        api_url=api_url,
        idx_col=idx_col,
        beg_page=beg_page,
        csv_name=csv_name,
        data_key='payload',
        status_key='status',
        on_page_key='page',
        of_pages_key='max_page',
        url_next_key='next_page',
        url_prev_key='previous_page',
        url_get_page='?page=',
        title=target_table,
        sep=sep,
        to_csv=to_csv,
        use_cache=use_cache,
        show_log=show_log,
        debug=debug,
        splain=splain,
        maxcols=maxcols,
    )
    
    return df


def get_web_csv_data(
    csv_name='opsd.csv', 
    csv_url='https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv', 
    use_cache=True, 
    splain=local_settings.splain,

    ):

    if use_cache and path.exists(csv_name):
        return check_df(pd.read_csv(csv_name), splain=splain)
    df = check_df(pd.read_csv(csv_url), splain=splain, title=csv_name)
    df.to_csv(csv_name, index=False)
    return df