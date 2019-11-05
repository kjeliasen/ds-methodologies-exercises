from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import sql_df
from dfo import DFO


sql = '''
select 
    p.`id`,
    p.`parcelid`,
    p.`airconditioningtypeid`,
    act.`airconditioningdesc`,
    p.`architecturalstyletypeid`,
    ast.`architecturalstyledesc`,
    p.`basementsqft`,
    p.`bathroomcnt`,
    p.`bedroomcnt`,
    p.`buildingclasstypeid`,
    bct.`buildingclassdesc`,
    p.`buildingqualitytypeid`,
    p.`calculatedbathnbr`,
    p.`calculatedfinishedsquarefeet`,
    p.`decktypeid`,
    p.`finishedfloor1squarefeet`,
    p.`finishedsquarefeet12`,
    p.`finishedsquarefeet13`,
    p.`finishedsquarefeet15`,
    p.`finishedsquarefeet50`,
    p.`finishedsquarefeet6`,
    p.`fips`,
    svi.`COUNTY` county,
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
--    p.`poolsizesum`,
--    p.`pooltypeid10`,
--    p.`pooltypeid2`,
--    p.`pooltypeid7`,
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
    p.`storytypeid`,
    st.`storydesc`,
    p.`taxvaluedollarcnt`,
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
    p.`typeconstructiontypeid`,
    tct.`typeconstructiondesc`,
    p.`censustractandblock`,
    pred.`transactiondate`,
    pred.`logerror`,
    m.`transactions`,
    p.`taxamount`/p.`taxvaluedollarcnt` tax_rate
from 
    `properties_2017` p
inner join `predictions_2017`  pred
    on p.`parcelid` = pred.`parcelid` 
inner join 
    (select 
        `parcelid`, 
        max(`transactiondate`) `lasttransactiondate`, 
        max(`id`) `maxid`, 
        count(*) `transactions`
    from 
        predictions_2017
    where
        transactiondate like '2017%%'
    group by 
        `parcelid`
    ) m
    on 
    pred.parcelid = m.parcelid
    and pred.transactiondate = m.lasttransactiondate

left join `propertylandusetype` plut
    on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
        
left join svi_db.svi2016_us_county svi
    on p.`fips` = svi.`FIPS`
left join `airconditioningtype` act
    using(`airconditioningtypeid`)
left join heatingorsystemtype hst
    using(`heatingorsystemtypeid`)
left join `architecturalstyletype` ast
    using(`architecturalstyletypeid`)
left join `buildingclasstype` bct
    using(`buildingclasstypeid`)
left join `storytype` st
    using(`storytypeid`)
left join `typeconstructiontype` tct
    using(`typeconstructiontypeid`)

where 
    p.`latitude` is not null
    and p.`longitude` is not null;
'''


@timeifdebug
def get_zillow_data(sql=sql, splain=local_settings.splain, **kwargs):
    '''
    get_zillow_data(splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    The output of this function passes through sql_df() and check_df().
    '''
    return sql_df(sql=sql, db='zillow', splain=splain, **kwargs)


def prep_zillow_data(dfo=None, df=None, splain=local_settings.splain, **kwargs):
    if dfo is None:
        dfo = set_dfo(get_zillow_data(), splain=splain)
    if df is None:
        df = dfo.df
    df = convert_to_dates(df, cols=['transactiondate'])
