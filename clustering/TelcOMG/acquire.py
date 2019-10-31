###############################################################################
### regression imports                                                      ###
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_profiling as pdspro


###############################################################################
### local imports                                                           ###
###############################################################################

from env import host, user, password
from debug import local_settings, timeifdebug, timeargsifdebug, frame_splain



###############################################################################
### get db url                                                              ###
###############################################################################

@timeifdebug  # <--- DO NOT RUN ARGS DEBUG HERE! Will pass password info.
def get_db_url(user=user, password=password, host=host, database='employees'):
    '''
    get_db_url(user=user, password=password, host=host, database='zillow')
    RETURNS login url for selected mysql database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'


###############################################################################
### classification functions                                                ###
###############################################################################

@timeifdebug
def paste_df(splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    return check_df(pd.read_clipboard(), splain=splain, **kwargs)


@timeifdebug
def excel_df(excel_path, splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    return check_df(pd.read_excel(excel_path), splain=splain, **kwargs)


@timeifdebug
def google_df(sheet_url, splain=local_settings.splain, **kwargs):
    '''
    fn
    RETURNS:
    '''
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return check_df(pd.read_csv(csv_export_url), splain=splain, **kwargs)


@timeargsifdebug
def csv_df(csv, splain=local_settings.splain, **kwargs):
    '''
    csv_df(csv, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    Reads a csv file into a dataframe, then sends the dataframe to check_df().
    '''
    csv = pd.read_csv(csv)
    return check_df(csv, splain=splain, **kwargs)


@timeargsifdebug
def sql_df(sql, db, splain=local_settings.splain, **kwargs):
    '''
    sql_df(sql, db, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    Reads a csv file into a dataframe, then sends the dataframe to check_df().
    '''
    db_url = get_db_url(database=db)
    return check_df(pd.read_sql(sql, db_url), splain=splain)

@timeifdebug
def check_df(dataframe, *args, splain=local_settings.splain, **kwargs):
    '''
    check_df(dataframe, splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    This function receives any dataframe, replaces null values with np.nan 
    and passes it through frame_splain(). If splain is true, frame_splain()
    will produce a report on the dataframe.
    '''

    df = dataframe.fillna(value=np.nan)#, inplace=True)
    frame_splain(df, splain=splain, **kwargs)
    return df


@timeargsifdebug
def get_telco_data(splain=local_settings.splain, **kwargs):
    '''
    get_telco_data(splain=local_settings.splain, **kwargs)
    RETURNS: dataframe

    The telco_churn dataset required for churn project. This function passes 
    through sql_df() and check_df().
    '''
    sql = '''
    select
        cust.`customer_id`,
        cust.`gender`,
        cust.`gender` = 'Male' is_male,
        cust.`gender` = 'Female' is_female,
        cust.`senior_citizen`,
        cust.`partner` = 'Yes' partner,
        cust.`dependents` = 'Yes' dependents,
        cust.`partner` = 'Yes' or cust.`dependents` = 'Yes' family,
        2 * case when cust.`partner` = 'Yes' then 1 else 0 end + case when cust.`dependents` = 'Yes' then 1 else 0 end partner_deps_id,
        concat(
            case when cust.`partner` = 'Yes' then 'Has ' else 'No ' end,
            'partner, ',
            case when cust.`dependents` = 'Yes' then 'has ' else 'no ' end,
            'dependents') partner_deps,
        cust.`tenure`,
        (cust.`tenure` DIV 12) tenure_years,
        case when cust.`contract_type_id` = 1 then 0 else (cust.`tenure` DIV case when cust.`contract_type_id` = 2 then 12 else 24 end) end contract_renews,
        case when cust.`contract_type_id` = 1 then 0 else (cust.`tenure` MOD case when cust.`contract_type_id` = 2 then 12 else 24 end) end remaining_months,
        cust.`tenure` > 1 thru_first_month,
        cust.`tenure` >= 3 thru_first_quarter,
        cust.`tenure` >= 6 thru_first_half,
        cust.`tenure` >= 12 thru_first_year,
        cust.`tenure` > case when cust.`contract_type_id` = 1 then 1 else case when cust.`contract_type_id` = 2 then 12 else 24 end end thru_first_term,
        cust.`phone_service` = 'Yes' phone_service,
        cust.`multiple_lines` = 'Yes' multiple_lines,
        case when cust.`phone_service` = 'Yes' then 1 + case when cust.`multiple_lines` = 'Yes' then 1 else 0 end else 0 end phone_service_id,
        case when cust.`phone_service` = 'Yes' then case when cust.`multiple_lines` = 'Yes' then 'Multiple Lines' else 'Single Line' end else 'No Phone' end phone_service_type,
        cust.`internet_service_type_id`,
        ist.`internet_service_type`,
        cust.`internet_service_type_id` <> 3 internet_service,
        cust.`internet_service_type_id` = 1 has_dsl,
        cust.`internet_service_type_id` = 2 has_fiber,
        cust.`online_security` = 'Yes' online_security,
        cust.`online_backup` = 'Yes' online_backup,
        cust.`online_security` = 'Yes' or cust.`online_backup` = 'Yes' online_security_backup,
        cust.`device_protection` = 'Yes' device_protection,
        cust.`tech_support` = 'Yes' tech_support,
        cust.`streaming_tv` = 'Yes' streaming_tv,
        cust.`streaming_movies` = 'Yes' streaming_movies,
        cust.`streaming_tv` = 'Yes' or `streaming_movies` = 'Yes' streaming_services,
        cust.`internet_service_type_id` = 1 and (cust.`streaming_tv` = 'Yes' or `streaming_movies` = 'Yes') streaming_dsl,
        cust.`internet_service_type_id` = 2 and (cust.`streaming_tv` = 'Yes' or `streaming_movies` = 'Yes') streaming_fiber,
        cust.`contract_type_id`,
        ct.`contract_type`,
        cust.`contract_type_id` <> 1 on_contract,
        case when cust.`contract_type_id` = 1 then 1 else case when cust.`contract_type_id` = 2 then 12 else 24 end end contract_duration,
        cust.`paperless_billing` = 'Yes' paperless_billing,
        cust.`payment_type_id`,
        pt.`payment_type`,
        pt.`payment_type` like '%%auto%%' auto_pay,
        pt.`payment_type` not like '%%auto%%' and cust.`contract_type_id` = 1 manual_mtm,
        cust.`monthly_charges`,
        case when cust.`total_charges` = '' then 0 else cast(cust.`total_charges` as decimal(11,2)) end total_charges,
        cast(case when cust.`tenure` = 0 then 0 else ((cast(cust.`total_charges` as decimal(11,2)) / cust.`tenure`) - cust.`monthly_charges`) end as decimal(11,2)) avg_monthly_variance,
        case when cust.`churn` = 'Yes' then 1 else 0 end churn
    from 
        customers cust
    left join 
        contract_types ct
        using(`contract_type_id`)
    left join 
        internet_service_types ist
        using(`internet_service_type_id`)
    left join 
        payment_types pt
        using(`payment_type_id`)
    '''
    return sql_df(sql=sql, db='telco_churn', splain=splain, **kwargs)


