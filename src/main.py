import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from datastory import DataStory
from datetime import datetime, date
from google.cloud import bigquery, secretmanager
from sklearn.linear_model import LinearRegression


if __name__=='__main__':
    Client = bigquery.Client('nais-analyse-prod-2dcc')
    df_bq = load_data_from_bq(Client, query_number=1)




def load_data_from_bq(Client, query_number):
    FIRST_YEAR = 2021
    FIRST_WEEK = 44 # 1. november 2021
    FIRST_MONTH = 11
    FIRST_FIN_YEAR = 2022 # Budget year nov-oct
    CURRENT_YEAR = datetime.now().year

    if query_number == 1:
        return Client.query("""
            select EXTRACT(year from dato) as year, EXTRACT(month from dato) as month,  EXTRACT(ISOWEEK from dato) as week,
            case when EXTRACT(month from dato) in (11, 12) then EXTRACT(year from dato) + 1
            else EXTRACT(year from dato) end as fin_year,
            env, service_description, sum(calculated_cost) as calculated_cost
            from `nais-analyse-prod-2dcc.nais_billing_nav.cost_breakdown_gcp`
            where dato >= "2021-11-01"
            and tenant in ("nav", "dev-nais", "example")
            group by year, month, week, fin_year, env, service_description
            order by year asc, month asc, week asc;
        """).result().to_dataframe()
    elif query_number == 2:
        return Client.query("""
            select dato, 
                case when env = "prod-gcp" then "prod"
                    when env = "dev-gcp" then "dev"
                    else env
                end as env, 
                service_description, 
                sum(calculated_cost) as calculated_cost 
            from `nais-analyse-prod-2dcc.nais_billing_nav.cost_breakdown_gcp`
            where current_date() - 22 <= dato and dato < current_date() - 1
            and tenant in ("nav", "dev-nais", "example")
            group by dato, env, service_description;
            """).result().to_dataframe()


def agg_week(row):
    if row.week >= 52 and row.month == 1:
        return row.week + 52 * (row.year - FIRST_YEAR - 1) - (FIRST_WEEK - 1)
    else:
        return row.week + 52 * (row.year - FIRST_YEAR) - (FIRST_WEEK - 1)
    
    
def unagg_week(week):
    week_num = (week + (FIRST_WEEK - 1)) % 52
    year = (week + (FIRST_WEEK - 1)) // 52 + FIRST_YEAR
    if week_num == 0:
        return f"{year-1} - {str(52).zfill(2)}"
    else:
        return f"{year} - {str(week_num).zfill(2)}"