###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import json

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
### web-scraping functions                                                  ###
###############################################################################

def get_val_from_key(dict, key='key', keys=None):
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
    to_csv=False,
    show_log=False,
    debug=False):
    
    # Setup df
    df = pd.DataFrame()

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
        payload = get_val_from_key(dict=response_json, key=data_key, keys=json_keys)
        if payload is None:
            break

        payload_keys = payload.keys()
        
        # Set navigation values
        on_page = get_val_from_key(dict=payload, key=on_page_key, keys=payload_keys)
        of_pages = get_val_from_key(dict=payload, key=of_pages_key, keys=payload_keys)
        url_next = get_val_from_key(dict=payload, key=url_next_key, keys=payload_keys)
        url_prev = get_val_from_key(dict=payload, key=url_prev_key, keys=payload_keys)
        
        # Get target data
        target_data = get_val_from_key(dict=payload, key=sect_key, keys=payload_keys)
        if target_data is None:
            break
        
        # Make page dataframe
        page_df = pd.DataFrame(target_data)
        if idx_col in page_df.columns:
            page_df.set_index(idx_col, inplace=True)
        else:
            print('index is missing')
            break
        
        df = df.append(page_df, verify_integrity=True)

        if url_next is None:
            keep_going = False
            
        if keep_going:
            next_url = url_next
        else:
            next_url = None
        
        is_complete = True
        
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
    to_csv=False,
    show_log=False,
    sect_url_keys={},
    sep=',',
    debug=False,
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
        sep=sep,
        to_csv=to_csv,
        show_log=show_log,
        debug=debug
    )
    
    return df
