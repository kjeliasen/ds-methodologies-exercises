from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain



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
def set_dfo(dfo_df, y_column=None, **kwargs):
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

