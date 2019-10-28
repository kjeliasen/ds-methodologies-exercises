co###############################################################################
### imports                                                                 ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

###############################################################################
### local imports                                                           ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain

###############################################################################
### get db url                                                              ###
###############################################################################

@timeifdebug  # <--- DO NOT RUN ARGS DEBUG HERE! Will pass password info.
def get_db_url(user, password, host, database):
    '''
    get_db_url(user=user, password=password, host=host, database='zillow')
    RETURNS login url for selected mysql database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'


###############################################################################
### other functions                                                         ###
###############################################################################

@timeargsifdebug
def get_sql(sql='zillow_sql'):
    '''
    get_sql(sql='zillow_sql')
    RETURNS sql[sql]

    Contains dictionary of sql statements that can be used in zillow dataset. 

    Pass an SQL key, returns the sql statement.
    '''
    sqls = {
        # test_sql for quick test of environment
        'test_sql': 'test',

        # zillow_sql to pull gross_props
        'zillow_sql': '''
        select 
            p.`bathroomcnt`,
            p.`bedroomcnt`,
            p.`calculatedfinishedsquarefeet`,

            p.`taxvaluedollarcnt`,
            
            svi.`COUNTY` county,
            p.`taxamount`/p.`taxvaluedollarcnt` tax_rate,

            p.`id`,
            p.`parcelid`,
            p.`airconditioningtypeid`,
            act.`airconditioningdesc`,
            p.`basementsqft`,
            p.`buildingqualitytypeid`,
            p.`calculatedbathnbr`,
            p.`decktypeid`,
            p.`finishedfloor1squarefeet`,
            p.`finishedsquarefeet12`,
            p.`finishedsquarefeet13`,
            p.`finishedsquarefeet15`,
            p.`finishedsquarefeet50`,
            p.`finishedsquarefeet6`,
            p.`fips`,
            svi.`ST_ABBR` state,
            p.`fireplacecnt`,
            p.`fullbathcnt`,
            p.`garagecarcnt`,
            p.`garagetotalsqft`,
            p.`hashottuborspa`,
            p.`heatingorsystemtypeid`,
            hst.`heatingorsystemdesc`,
            p.`latitude`,
            p.`longitude`,
            p.`lotsizesquarefeet`,
            p.`poolcnt`,
            p.`poolsizesum`,
            p.`pooltypeid10`,
            p.`pooltypeid2`,
            p.`pooltypeid7`,
            p.`propertycountylandusecode`,
            p.`propertylandusetypeid`,
            plut.`propertylandusedesc`,
            p.`propertyzoningdesc`,
            p.`rawcensustractandblock`,
            p.`regionidcity`,
            p.`regionidcounty`,
            p.`regionidneighborhood`,
            p.`regionidzip`,
            p.`roomcnt`,
            p.`threequarterbathnbr`,
            p.`unitcnt`,
            p.`yardbuildingsqft17`,
            p.`yardbuildingsqft26`,
            p.`yearbuilt`,
            p.`numberofstories`,
            p.`fireplaceflag`,
            p.`structuretaxvaluedollarcnt`,
            p.`assessmentyear`,
            p.`landtaxvaluedollarcnt`,
            p.`taxamount`,
            p.`taxdelinquencyflag`,
            p.`taxdelinquencyyear`, 
            p.`censustractandblock`,
            pred.`transactiondate`

        from 
            `properties_2017` p

        inner join `predictions_2017`  pred
            on p.`parcelid` = pred.`parcelid` 
            and (transactiondate like '2017-05%%' 
            or transactiondate like '2017_06%%')

        inner join `propertylandusetype` plut
            on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
            and plut.`propertylandusetypeid` not in (31, 246, 247, 248)	
            
        left join svi_db.svi2016_us_county svi
            on p.`fips` = svi.`FIPS`

        left join `airconditioningtype` act
            using(`airconditioningtypeid`)

        left join heatingorsystemtype hst
            using(`heatingorsystemtypeid`)

        where 
            taxvaluedollarcnt is not null		
            and p.`calculatedfinishedsquarefeet` is not null	
            and p.`lotsizesquarefeet` is not null
            and p.`bathroomcnt` * p.`bedroomcnt` <> 0
            and p.`taxamount` is not null
        ;'''
    }

    return sqls[sql]


@timeargsifdebug
def wrangle_zillow(db='zillow', sql='zillow_sql', sql_string=False):
    '''
    wrangle_zillow(db='zillow', sql='zillow_sql', sql_string=False)
    RETURNS result_df

    Pass database name ('zillow' by default) and either preset sql with 
    sql_string=False (default) or sql statement with sql_string=True.

    Produces results of SQL statement in a dataframe object.

    *** Requires user, password, and host from env.py ***
    '''
    get_database = db
    zillow_url = get_db_url(user=user, password=password, host=host, database=get_database)
    use_sql = sql if sql_string else get_sql(sql='zillow_sql')
    result_df = pd.read_sql(use_sql, zillow_url)
    frame_splain(result_df, topx=5, maxcols=10)
    return result_df    


