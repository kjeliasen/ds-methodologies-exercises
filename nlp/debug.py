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
            t1 = datetime.now()
            print(get_date_time_code(datetime=t1, in_format='%Y-%m-%d %H:%M:%S'), 'starting', fn.__name__)
        result = fn(*args, **kwargs)
        if local_settings.debug:
            t2 = datetime.now()
            print(get_date_time_code(datetime=t2, in_format='%Y-%m-%d %H:%M:%S'), 'ending', fn.__name__, '; time:', t2 - t1)
        return result
    return inner


def timeargsifdebug(fn):
    '''
    fn
    RETURNS:
    '''
    def inner(*args, **kwargs):
        if local_settings.debug:
            t1 = datetime.now()
            print(get_date_time_code(datetime=t1, in_format='%Y-%m-%d %H:%M:%S'), 'starting', fn.__name__)
            if args:
                print('Args:', args)
            if kwargs:
                print('Kwargs:', kwargs)
        result = fn(*args, **kwargs)
        if local_settings.debug:
            t2 = datetime.now()
            print(get_date_time_code(datetime=t2, in_format='%Y-%m-%d %H:%M:%S'), 'ending', fn.__name__, '; time:', t2 - t1)
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

    if splain:

        cols = df.shape[1]
        max_x = min(topx, df.shape[0])
        print(title.upper(), 'SHAPE:')
        print(df.shape)
        print()
        print(title.upper(), 'INFO:')
        print(df.info())
        print()
        print(title.upper(), 'DESCRIPTION:')
        print(df.describe().transpose())
        print()
        print(title.upper(), 'HEAD:')
        if cols <= maxcols:
            print(df.head(max_x))
        else:
            print(df.head().T)
            print()


def get_date_time_code(datetime=datetime.now(), in_format='%Y%d%m%H%M'):
    return datetime.strftime(in_format)
