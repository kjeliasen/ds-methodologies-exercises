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
    cols = df.shape[1]
    max_x = min(topx, df.shape[0])
    if splain:
        print(title.upper(), 'SHAPE:')
        print(df.shape, '\n')
        print(title.upper(), 'INFO:')
        print(df.info(), '\n')
        print(title.upper(), 'DESCRIPTION:')
        print(df.describe().transpose(), '\n')
        if cols <= maxcols:
            print(title.upper(), 'HEAD:')
            print(df.head(max_x), '\n')
