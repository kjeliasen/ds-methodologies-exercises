from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain
from acquire import sql_df, csv_df, check_df
from dfo import DFO


sqls = {
'full': '''
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
''', 
'mvp': '''
select 
    p.`id`,
    p.`parcelid`,
--    p.`airconditioningtypeid`,
--    act.`airconditioningdesc`,
--    p.`architecturalstyletypeid`,
--    ast.`architecturalstyledesc`,
--    p.`basementsqft`,
    p.`bathroomcnt`,
    p.`bedroomcnt`,
--    p.`buildingclasstypeid`,
--    bct.`buildingclassdesc`,
--    p.`buildingqualitytypeid`,
--    p.`calculatedbathnbr`,
    p.`calculatedfinishedsquarefeet`,
--    p.`decktypeid`,
--    p.`finishedfloor1squarefeet`,
--    p.`finishedsquarefeet12`,
--    p.`finishedsquarefeet13`,
--    p.`finishedsquarefeet15`,
--    p.`finishedsquarefeet50`,
--    p.`finishedsquarefeet6`,
    p.`fips`,
--    svi.`COUNTY` county,
--    svi.`ST_ABBR` state,
--    p.`fireplacecnt`,
--    p.`fullbathcnt`,
--    p.`garagecarcnt`,
--    p.`garagetotalsqft`,
--    p.`hashottuborspa`,
--    p.`heatingorsystemtypeid`,
--    hst.`heatingorsystemdesc`,
    p.`latitude`,
    p.`longitude`,
--    p.`lotsizesquarefeet`,
--    p.`poolcnt`,
--    p.`poolsizesum`,
--    p.`pooltypeid10`,
--    p.`pooltypeid2`,
--    p.`pooltypeid7`,
--    p.`propertycountylandusecode`,
--    p.`propertylandusetypeid`,
--    plut.`propertylandusedesc`,
--    p.`propertyzoningdesc`,
--    p.`rawcensustractandblock`,
--    p.`regionidcity`,
--    p.`regionidcounty`,
--    p.`regionidneighborhood`,
--    p.`regionidzip`,
--    p.`roomcnt`,
--    p.`storytypeid`,
--    st.`storydesc`,
--    p.`taxvaluedollarcnt`,
--    p.`threequarterbathnbr`,
--    p.`unitcnt`,
--    p.`yardbuildingsqft17`,
--    p.`yardbuildingsqft26`,
--    p.`yearbuilt`,
--    p.`numberofstories`,
--    p.`fireplaceflag`,
    p.`structuretaxvaluedollarcnt`,
--    p.`assessmentyear`,
--    p.`landtaxvaluedollarcnt`,
--    p.`taxamount`,
--    p.`taxdelinquencyflag`,
--    p.`taxdelinquencyyear`, 
--    p.`typeconstructiontypeid`,
--    tct.`typeconstructiondesc`,
--    p.`censustractandblock`,
--    pred.`transactiondate`,
--    m.`transactions`,
--    p.`taxamount`/p.`taxvaluedollarcnt` tax_rate
    (p.`structuretaxvaluedollarcnt` / p.`calculatedfinishedsquarefeet`) structuredollarpersqft,
    pred.`logerror`
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

-- left join `propertylandusetype` plut
--     on p.`propertylandusetypeid` = plut.`propertylandusetypeid`
-- left join svi_db.svi2016_us_county svi
--     on p.`fips` = svi.`FIPS`
-- left join `airconditioningtype` act
--     using(`airconditioningtypeid`)
-- left join heatingorsystemtype hst
--     using(`heatingorsystemtypeid`)
-- left join `architecturalstyletype` ast
--     using(`architecturalstyletypeid`)
-- left join `buildingclasstype` bct
--     using(`buildingclasstypeid`)
-- left join `storytype` st
--     using(`storytypeid`)
-- left join `typeconstructiontype` tct
--     using(`typeconstructiontypeid`)

where 
    p.`propertylandusetypeid` = 261
    and p.`bathroomcnt` * p.`bedroomcnt` <> 0
    and p.`calculatedfinishedsquarefeet` is not null
    and p.`calculatedfinishedsquarefeet` > 0
    and p.`structuretaxvaluedollarcnt` is not null
    and p.`latitude` is not null
    and p.`longitude` is not null;
'''
}


keep_cols={
    'mvp': [
        'parcelid',
        'bathroomcnt', 
        'bedroomcnt', 
        'calculatedfinishedsquarefeet', 
        'latitude', 
        'longitude', 
        'structuretaxvaluedollarcnt', 
        'structuredollarpersqft', 
        'logerror'
    ]
}


@timeifdebug
def get_zillow_data(sql=sqls['full'], db='zillow', splain=local_settings.splain, **kwargs):
    '''
    get_zillow_data(splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    The output of this function passes through sql_df() and check_df().
    '''
    return sql_df(sql=sql, db=db, splain=splain, **kwargs)


def prep_zillow_data(dfo=None, df=None, splain=local_settings.splain, **kwargs):
    if dfo is None:
        dfo = set_dfo(get_zillow_data(), splain=splain)
        df = dfo.df
    if df is None:
        df = dfo.df
    df = convert_to_dates(df, cols=['transactiondate'])


@timeifdebug
def refresh_zillow_csv(sql=sqls['mvp'], db='zillow', output_csv='zillow_local_mvp.csv', sep='|', splain=local_settings.splain, **kwargs):
    df = get_zillow_data(sql=sql, db=db, splain=splain)
    df = df.set_index('id')
    df.to_csv(path_or_buf=output_csv, 
        sep=sep, 
        na_rep='', 
        float_format=None, 
        columns=None, 
        header=True, 
        index=True, 
        index_label='id', 
        mode='w', 
        encoding=None, 
        compression='infer', 
        quoting=None, 
        quotechar='"', 
        line_terminator=None, 
        chunksize=None, 
        date_format=None, 
        doublequote=True, 
        escapechar=None, 
        decimal='.')
    
    
@timeifdebug    
def get_zillow_local_data(csv='zillow_local.csv', splain=local_settings.splain, sep='|', **kwargs):    
    return csv_df(csv, sep=sep, splain=local_settings.splain, **kwargs)


@timeifdebug
def reduce_df_to_cols(df, column_list=keep_cols['mvp'], csv='zillow_local.csv', sep='|', **kwargs):
    get_zillow_local_data(csv=csv, sep=sep, **kwargs)
    col_list = [col for col in column_list if col in df.columns]
    new_df = df[col_list]

    return check_df(new_df, splain=local_settings.splain, **kwargs)


if __name__ == '__main__':
    
    local_settings.debug = True
    local_settings.splain = True
    
    refresh_zillow_csv()
    #reduce_df_to_cols(get_zillow_local_data(splain=False))