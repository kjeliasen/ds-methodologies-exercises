import time
import datetime

from datetime import datetime
# local_settings = {}
# local_settings = {'DEBUG': False, 'ARGS': False, 'KWARGS': False, 'SPLAIN': False, 'TOPX': 5, 'MAXCOLS': 10}

class Settings(): pass


def set_settings():
    '''
    fn
    RETURNS:
    '''
    use_settings = Settings()
    use_settings.debug = False
    use_settings.see_args = False
    use_settings.see_kwargs = False
    use_settings.splain = False
    use_settings.topx = 5
    use_settings.maxcols = 10
    return use_settings


local_settings = set_settings()


def timeifdebug(fn):
    '''
    fn
    RETURNS:
    '''
    def inner(*args, **kwargs):
        if local_settings.debug:
            print(get_date_time_code(format='%Y-%m-%d %H:%M:%S'), 'starting', fn.__name__)
            t1 = time.time()
        result = fn(*args, **kwargs)
        if local_settings.debug:
            print(get_date_time_code(format='%Y-%m-%d %H:%M:%S'), 'ending', fn.__name__, '; time:', time.time() - t1)
        return result
    return inner


def timeargsifdebug(fn):
    '''
    fn
    RETURNS:
    '''
    def inner(*args, **kwargs):
        if local_settings.debug:
            print(get_date_time_code(format='%Y-%m-%d %H:%M:%S'), 'starting', fn.__name__)
            if args:
                print('Args:', args)
            if kwargs:
                print('Kwargs:', kwargs)
            t1 = time.time()
        result = fn(*args, **kwargs)
        if local_settings.debug:
            print(get_date_time_code(format='%Y-%m-%d %H:%M:%S'), 'ending', fn.__name__, '; time:', time.time() - t1)
        return result
    return inner


@timeifdebug
def frame_splain(
        df, 
        title='DATAFRAME', 
        *args, 
        topx=local_settings.topx, 
        maxcols=local_settings.maxcols, 
        splain=local_settings.splain, 
        **kwargs
    ):
    '''
    frame_splain(
        df, 
        title='DATAFRAME', 
        *args, 
        topx=local_settings.topx, 
        maxcols=local_settings.maxcols, 
        splain=local_settings.splain, 
        **kwargs
    )
    RETURNS: Summary data of a dataframe, including shape, info, description, 
    and the top topx rows if column count is not more than maxcols
    '''
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


@timeargsifdebug
def get_date_time_code(datetime=datetime.now(), in_format='%Y%d%m%H%M'):
    return datetime.strftime(in_format)
