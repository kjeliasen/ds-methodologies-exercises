from prepare import split_my_data, xy_df, standard_scaler



###############################################################################
### establish DFO class                                                     ###
###############################################################################

class DFO(): 
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


###############################################################################
### DFO functions                                                           ###
###############################################################################

@timeifdebug
def set_dfo(dfo_df, y_column, train_pct=.75, randomer=None, **kwargs):
    '''
    set_dfo(dfo_df=get_base_df(), y_column='taxable_value', train_pct=.75, randomer=None, scaler_fn=standard_scaler, **kwargs)
    RETURNS: dfo object with heaping piles of context enclosed
    
    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing
    '''
    dfo = DFO()
    dfo.df = dfo_df
    dfo.y_column = y_column
    return dfo


def scale_dfo(dfo, scaler_fn=standard_scaler):
    '''
    scale_dfo(dfo, scaler_fn=standard_scaler)
    RETURNS: dfo object with heaping piles of context enclosed
    
    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing
    '''
    dfo.scaler, dfo.train_scaled, dfo.test_scaled = scaler_fn(train=dfo.train, test=dfo.test)
    return dfo


def split_dfo(dfo, train_pct=.75, randomer=None):
    '''
    scale_dfo(dfo, scaler_fn=standard_scaler)
    RETURNS: dfo object with heaping piles of context enclosed
    
    scaler_fn must be a function
    dummy val added to train and test to allow for later feature selection testing
    '''
    dfo.randomer = randomer
    dfo.train, dfo.test = split_my_data(df=target_df, random_state=randomer)
    dfo.train['dummy_val']=1
    dfo.train_scaled['dummy_val']=1
    dfo.X_train, dfo.y_train = xy_df(dataframe=dfo.train, y_column=y_column)
    dfo.X_test, dfo.y_test = xy_df(dataframe=dfo.test, y_column=y_column)
    dfo.X_train_scaled, dfo.y_train_scaled = xy_df(dataframe=dfo.train_scaled, y_column=y_column)
    dfo.X_test_scaled, dfo.y_test_scaled = xy_df(dataframe=dfo.test_scaled, y_column=y_column)
    return dfo
