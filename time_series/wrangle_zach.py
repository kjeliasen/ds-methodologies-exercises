from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import output_payload_data, check_df

base_url = 'https://python.zach.lol'
sect_urls = {
    'items': {
        'sect_key': 'items', 
        'api_url': '/api/v1/items',
        'idx_col': 'item_id', 
        'page_beg': 1,
        'csv_name': 'items.csv'
    },
    'stores': {
        'sect_key': 'stores', 
        'api_url': '/api/v1/stores', 
        'idx_col': 'store_id', 
        'page_beg': 1, 
        'csv_name': 'stores.csv'
    },
    'sales': {
        'sect_key': 'sales', 
        'api_url': '/api/v1/sales', 
        'idx_col': 'sale_id', 
        'page_beg': 1, 
        'csv_name': 'sales.csv'
    },
}

def set_combo_data(splain=True, splain_inner=False, maxcols=16, debug=False):

    items_df = output_payload_data(
        target_table='items', 
        base_url=base_url,
        show_log=False,
        sect_url_keys=sect_urls,
        sep=',',
        debug=debug, 
        use_cache=True,
        to_csv=False,
        splain=splain_inner
        )

    stores_df = output_payload_data(
        target_table='stores', 
        base_url=base_url,
        show_log=False,
        sect_url_keys=sect_urls,
        sep=',',
        debug=False, 
        use_cache=True,
        to_csv=False,
        splain=splain_inner
        )
    
    sales_df = output_payload_data(
        target_table='sales', 
        base_url=base_url,
        show_log=False,
        sect_url_keys=sect_urls,
        sep=',',
        debug=False, 
        use_cache=True,
        to_csv=False,
        splain=splain_inner
        )
    
    combo_df = (sales_df
        .merge(
            items_df, 
            how='left', 
            left_on='item', 
            right_on='item_id', 
            validate='m:1'
            )
        .merge(
            stores_df, 
            how='left', 
            left_on='store', 
            right_on='store_id', 
            validate='m:1'
            )
        .drop(columns=['store_id','item_id'])
        .rename(columns={'store': 'store_id', 'item': 'item_id'})
        )

    df = check_df(combo_df, splain=splain, title='combo')

    return df
    
if __name__ == '__main__':
    df=set_combo_data(splain=True, splain_inner=False, debug=True)
    print(df.head(15))