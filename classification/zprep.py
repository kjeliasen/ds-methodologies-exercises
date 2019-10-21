###############################################################################
### imports                                                                 ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain


_global_renames = (
    ('parcelid', 'pid'),
    ('bathroomcnt', 'nbr_bedrms'),
    ('bedroomcnt', 'nbr_bthrms'),
    ('calculatedfinishedsquarefeet', 'finished_sqft'),
    ('taxvaluedollarcnt', 'taxable_value'),
)

            # ('county',''),
            # ('tax_rate',''),

            # ('id',''),
            # ('parcelid',''),
            # ('airconditioningtypeid','aircond_type'),
            # ('airconditioningdesc','aircond_desc'),
            # ('basementsqft','basement_sqft'),
            # ('buildingqualitytypeid','bldgqual_type'),
            # ('calculatedbathnbr',''),
            # ('decktypeid',''),
            # ('finishedfloor1squarefeet',''),
            # ('finishedsquarefeet12',''),
            # ('finishedsquarefeet13',''),
            # ('finishedsquarefeet15',''),
            # ('finishedsquarefeet50',''),
            # ('finishedsquarefeet6',''),
            # ('fireplacecnt',''),
            # ('fullbathcnt',''),
            # ('garagecarcnt',''),
            # ('garagetotalsqft',''),
            # ('hashottuborspa',''),
            # ('heatingorsystemtypeid',''),
            # ('heatingorsystemdesc',''),
            # ('latitude',''),
            # ('longitude',''),
            # ('lotsizesquarefeet',''),
            # ('poolcnt',''),
            # ('poolsizesum',''),
            # ('pooltypeid10',''),
            # ('pooltypeid2',''),
            # ('pooltypeid7',''),
            # ('propertycountylandusecode',''),
            # ('propertylandusetypeid',''),
            # ('propertylandusedesc',''),
            # ('propertyzoningdesc',''),
            # ('rawcensustractandblock',''),
            # ('regionidcity',''),
            # ('regionidcounty',''),
            # ('regionidneighborhood',''),
            # ('regionidzip',''),
            # ('roomcnt',''),
            # ('threequarterbathnbr',''),
            # ('unitcnt',''),
            # ('yardbuildingsqft17',''),
            # ('yardbuildingsqft26',''),
            # ('yearbuilt',''),
            # ('numberofstories',''),
            # ('fireplaceflag',''),
            # ('structuretaxvaluedollarcnt',''),
            # ('assessmentyear',''),
            # ('landtaxvaluedollarcnt',''),
            # ('taxamount',''),
            # ('taxdelinquencyflag',''),
            # ('taxdelinquencyyear',''), 
            # ('censustractandblock',''),
            # ('transactiondate','')

###############################################################################
### get data from acquire                                                   ###
###############################################################################


@timeifdebug
def edit_gross_df(dataframe):
    '''
    get_acquire_df(dataframe)
    RETURNS dataframe with columns renamed

    Fields will be renamed according to rules set in rename_fields()
    '''
    return rename_fields(dataframe)


@timeifdebug
def rename_fields(dataframe):
    '''
    rename_fields(dataframe)
    
    '''
    columns = dataframe.columns.tolist()
    renames = {k: v for k, v in _global_renames if k in columns}
    renamed_df = dataframe.rename(columns=renames)
    frame_splain(renamed_df, title='renamed df')
    return renamed_df


@timeifdebug
def edit_prep_df(dataframe):
    '''
    set_base_df(dataframe)
    RETURN prepped_df

    Gets basic dataframe for MVP objective. Features include bathrooms, 
    bedrooms, and square footage. Target variable is 'taxable_value'
    '''

    keep_fields = ['nbr_bthrms','nbr_bedrms','finished_sqft','taxable_value']
    prepped_df = dataframe[keep_fields]

    frame_splain(prepped_df, title='prepped df')
    return prepped_df


# print('Got Prep')