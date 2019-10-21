import time

# local_settings = {}
local_settings = {'DEBUG': False, 'ARGS': False, 'KWARGS': False, 'SPLAIN': False, 'TOPX': 5, 'MAXCOLS': 10}


def timeifdebug(fn):
    def inner(*args, **kwargs):
        if local_settings['DEBUG']:
            print('starting', fn.__name__)
            t1 = time.time()
        result = fn(*args, **kwargs)
        if local_settings['DEBUG']:
            print('ending', fn.__name__, '; time:', time.time() - t1)
        return result
    return inner


def timeargsifdebug(fn):
    def inner(*args, **kwargs):
        if local_settings['DEBUG']:
            print(fn.__name__, args, kwargs)
            t1 = time.time()
        result = fn(*args, **kwargs)
        if local_settings['DEBUG']:
            print('ending', fn.__name__, '; time:', time.time() - t1)
        return result
    return inner


def frame_splain(df, title='DATAFRAME', topx=local_settings['TOPX'], maxcols=local_settings['MAXCOLS'], splain=local_settings['SPLAIN']):
    df_shape = df.shape
    cols = df_shape[0]
    max_x = min(topx, df_shape[1])
    df_desc = df.describe()
    df_head = df.head(max_x)
    df_info = df.info()
    if splain:
        print(title, 'shape:\n', df_shape, '\n')
        print(title, 'info:\n', df_info, '\n')
        if cols <= maxcols:
            print(title, 'description:\n', df_desc, '\n')
            print(title, 'head:\n', df_head, '\n')
        
