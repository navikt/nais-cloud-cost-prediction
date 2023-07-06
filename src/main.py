import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from datastory import DataStory
from datetime import datetime, date, timedelta
from google.cloud import bigquery, secretmanager
from sklearn.linear_model import LinearRegression


def load_data_from_bq(Client, query_number):
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
            SELECT month, env, service_description, sum(calculated_cost) as calculated_cost, status,
            case when extract(month from month) in (11,12) then extract(year from month) + 1
            else extract(year from month) end as fin_year
            from `nais-analyse-prod-2dcc.nais_billing_nav.cost_breakdown_aiven`
            group by month, env, service_description, status
            order by month
            """).result().to_dataframe()
    elif query_number == 3:
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
    else:
        print(f"No data loaded from BigQuery because query_number {query_number} does not exist")


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


def prepare_df_service(df):
    df["n_week"] = df.apply(lambda row: agg_week(row), axis=1)
    cols = ["week", "n_week", "service_description"]
    current_n_week = df.n_week.max()
    df_service = df[df.n_week < current_n_week].groupby(cols, as_index=False).calculated_cost.sum().sort_values("n_week").reset_index(drop=True)
    df_service["year_week"] = df_service.n_week.apply(unagg_week)
   
    return df_service, current_n_week


def make_services_fig(df_service):
    # 'Cloud Dialogflow API' ødelegger rekkefølgen på stolpene så vi tar den bort. Det er snakk om mindre enn en euro.
    df_plot_services = df_service[df_service.service_description != 'Cloud Dialogflow API']
    fig = px.bar(df_plot_services, "year_week", "calculated_cost", color="service_description", barmode="stack", labels={"year_week":"År - uke", "calculated_cost":"Kostnad (€)"})
    fig.update_xaxes(categoryorder='array', categoryarray=df_service.year_week.unique() )
    return fig


def group_services(service, top_services):
    if service in top_services:
        return service
    else:
        return "Other services"


def make_training_data(df_service, current_n_week, weeks_to_look_at, n_top=2):
    top_services = set(df_service
                    .groupby("service_description", as_index=False)
                    .calculated_cost
                    .sum()
                    .sort_values("calculated_cost", ascending=False)
                    .iloc[:n_top]
                    .service_description
                    )
    df = df_service.copy()
    df["service_description"] = df.service_description.apply(lambda x: group_services(x, top_services))
    df_s = df.groupby(["service_description", "n_week"], as_index=False).calculated_cost.sum()

    start_n_week = current_n_week - weeks_to_look_at 
    df_s = df_s[(df_s.n_week >= start_n_week) & (df_s.n_week < current_n_week)]

    x_s = df_s[["service_description", "n_week"]]
    x_s = pd.get_dummies(data=x_s)
    y_s = df_s[["calculated_cost"]]

    return x_s, y_s


def make_week_vars(current_n_week, x_s):
    previous_n_week = current_n_week - 1
    n_weeks = 52*2 - previous_n_week
    n_services = x_s.shape[1] - 1
    return previous_n_week, n_weeks, n_services


def predict_cost(mod, current_n_week, n_weeks, n_services):
    last_week = 2*52+1
    pred = np.zeros([n_weeks + 1, n_services])
    for week in range(current_n_week, 2*52+1):
        for service in range(n_services):
            z = np.zeros(n_services, dtype=np.int64)
            z[service] = 1
            x1 = np.append(week, z).reshape(1,n_services+1)
            pred[week-current_n_week, service] = mod.predict(x1)[0][0]
    # Last week of October 2023 has only two days.
            if week == last_week:
                pred[week-current_n_week, service] *= 2/7
    return pred


def pred_to_df(pred, x_s, current_n_week):
    df_pred = pd.DataFrame(pred)
    df_pred.columns = pd.Series(x_s.columns[1:]).apply(lambda x: x[20:])

    df_pred["n_week"] = df_pred.index + current_n_week
    return df_pred


def prepare_df_plot(df_pred, current_n_week):
    df_plot = df_pred.sum(axis=1).reset_index(name="cost")
    df_plot["year_week"] = df_plot["index"] + current_n_week
    df_plot["year_week"] = df_plot["year_week"].apply(unagg_week)
    df_plot["index"] = df_plot["index"] + current_n_week - 52
    df_plot = df_plot.rename(columns={"index":"week"})
    return df_plot


def compute_total_costs(df_bq, df_pred, current_n_week):
    kostnad_i_fjor = df_bq[df_bq.fin_year == 2022].calculated_cost.sum()
    kostnad_hittil = df_bq[(df_bq.fin_year == 2023) & (df_bq.n_week < current_n_week)].calculated_cost.sum()

    forventet_resten = df_pred.sum(axis=1).sum()
    forventet_totalt = kostnad_hittil + forventet_resten
    return kostnad_i_fjor, kostnad_hittil, forventet_resten, forventet_totalt


def make_numbers_fig(kostnad_i_fjor, kostnad_hittil, forventet_resten, forventet_totalt):
    fig_numbers = sp.make_subplots(rows=2, cols=2,
                                 specs=[[{"type": "indicator"} ,{"type": "indicator"}],
                                       [{"type": "indicator"},{"type": "indicator"}]])
    fig_numbers.add_trace(go.Indicator(
            mode = 'number', 
            value = int(kostnad_hittil),
            number = {'valueformat': ',', 'suffix' :' €'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Kostnad hittil i år (eksludert inneværende uke)")}
        ), row = 1, col = 1)
    fig_numbers.add_trace(go.Indicator(
            mode = 'number', 
            value = int(forventet_resten),
            number = {'valueformat': ',', 'suffix' :' €'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Forventet kostnad resten av året")}
        ), row = 2, col = 1)
    fig_numbers.add_trace(go.Indicator(
            mode = 'number', 
            value = int(forventet_totalt),
            number = {'valueformat': ',', 'suffix' :' €'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Forventet totalkostnad i år")}
        ), row = 1, col = 2)
    fig_numbers.add_trace(go.Indicator(
            mode = 'number', 
            value = int(kostnad_i_fjor),
            number = {'valueformat': ',', 'suffix' :' €'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Totalkostnad i fjor")}
        ), row = 2, col = 2)
    return fig_numbers


def make_forecast_fig(df_service, df_pred, current_n_week):
    df_tot = df_service.groupby(["n_week"], as_index=False).calculated_cost.sum()
    df_tot["year_week"] = df_tot.n_week.apply(unagg_week)
    df_plot = prepare_df_plot(df_pred, current_n_week)

    fig_forecast = px.bar(df_tot, "year_week", "calculated_cost", title="Kostnad per uke november 2021 - oktober 2023", color_discrete_sequence=["rgba(0, 86, 180, 1)"], labels={"year_week":"Uke", "calculated_cost":"Kostnad (€)"})
    fig_forecast.update_traces(name='Faktisk kostnad', showlegend = True)
    fig_forecast.add_vline(x='23 - 01', line_dash="dash", line_color="gray")
    fig_forecast.add_traces(go.Bar(x=df_plot.year_week, y=df_plot.cost, name="Forventet kostnad", marker_color="rgba(153, 195, 255, 1)"))
    return fig_forecast


def prediction_history(current_n_week, df_service, weeks_to_look_at):
    first_n_week = 53
    predictions = np.zeros([current_n_week - first_n_week + 1])

    for n_week in range(first_n_week, current_n_week + 1):
        x, y = make_training_data(df_service, n_week, weeks_to_look_at, 2)
        mod = LinearRegression().fit(x.values, y.values)
        previous_n_week, n_weeks, n_services = make_week_vars(n_week, x)
        pred = predict_cost(mod, n_week, n_weeks, n_services)
        df_pred = pred_to_df(pred, x, n_week)
        kostnad_i_fjor, kostnad_hittil, forventet_resten, forventet_totalt = compute_total_costs(df_bq, df_pred, n_week)
        #print(f"Kostnad i fjor: {int(kostnad_i_fjor)} € \nKostnad hittil i år: {int(kostnad_hittil)} € \nForventet gjenstående kostnad i år: {int(forventet_resten)} € \nForventet totalkostnad i år: {int(forventet_totalt)} €")
        predictions[n_week - first_n_week] = forventet_totalt

    fig = px.line(x=np.linspace(predictions.shape[0]-1, 0, predictions.shape[0]), y=predictions,
              title="Totalkostnad i 2023 predikert på ulike tidspunkt",
              labels={"x":"Uker siden", "y":"Forventet kostnad 2023 (€)"})
    fig.update_xaxes(autorange="reversed")
    return fig


def make_aiven_services_fig(df_aiven):
    df = df_aiven.groupby(["month", "service_description"], as_index=False).calculated_cost.sum()
    df = df[df.month >= date(2021, 11, 1)]
    fig = px.bar(df, "month", "calculated_cost", color='service_description', 
        labels={"month":"Måned", "calculated_cost":"Kostnad ($)", "service_description":"Service"})
    return fig


def agg_month(x):
    return x.month + 2 + (x.year - (FIRST_FIN_YEAR)) * 12


def unagg_month(x):
    year = FIRST_YEAR + (x + 9) // 12
    month = (x + 9) % 12 + 1
    return date(year, month, 1) 


def group_aiven_month(df_aiven):
    df = (df_aiven
        [(df_aiven.status != 'estimate') & (df_aiven.fin_year >= 2022)]
        .groupby(["month", "fin_year"], as_index=False)
        .calculated_cost
        .sum()
        )
    df["n_month"] = df.month.apply(lambda x: agg_month(x))
    return df


def make_month_vars(df_aiven_month):
    previous_n_month = df_aiven_month.n_month.max()
    current_n_month = previous_n_month + 1
    n_months = 2 * 12 - previous_n_month
    return current_n_month, n_months


def make_training_data_aiven(df_aiven_month, current_n_month, months_to_look_at):
    start_n_month = current_n_month - months_to_look_at
    df = df_aiven_month.query(f"{start_n_month} <= n_month  & n_month < {current_n_month}")
    x = df.n_month
    y = df.calculated_cost
    return x, y


def linear_regression_aiven(x, y):
    return LinearRegression().fit(x.values.reshape(-1,1), y.values)


def predict_aiven(mod, n_months, current_n_month):
    pred = np.zeros(n_months)
    for month in range(current_n_month, 2*12+1):
        x1 = np.reshape(month, (1, 1))
        pred[month - current_n_month] = mod.predict(x1)[0]
    
    df = pd.DataFrame(pred)
    df.columns = pd.Series("predicted_cost")
    df["n_month"] = df.index + current_n_month
    df["month"] = df.n_month.apply(lambda x: unagg_month(x))

    return df


def make_forecast_fig_aiven(df_aiven_month, df_pred_aiven):
    fig = px.bar(df_aiven_month, "month", "calculated_cost", title="Aivenkostnad per måned for november 2021 - oktober 2023", color_discrete_sequence=["rgba(0, 86, 180, 1)"], labels={"month":"Måned", "calculated_cost":"Kostnad ($)"})
    fig.update_traces(name='Faktisk kostnad', showlegend = True)
    fig.add_traces(go.Bar(x=df_pred_aiven.month, y=df_pred_aiven.predicted_cost, name="Forventet kostnad", marker_color="rgba(153, 195, 255, 1)"))
    return fig


def compute_total_costs_aiven(df_aiven, df_pred_aiven):
    ifjor = df_aiven.query("fin_year == 2022").calculated_cost.sum()
    hittil = df_aiven.query("fin_year == 2023 & status != 'estimated'").calculated_cost.sum()
    resten = df_pred_aiven.predicted_cost.sum()
    totalt = float(hittil) + resten
    return ifjor, hittil, resten, totalt


def make_numbers_fig_aiven(ifjor, hittil, resten, totalt):
    fig = sp.make_subplots(rows=2, cols=2,
                                 specs=[[{"type": "indicator"} ,{"type": "indicator"}],
                                       [{"type": "indicator"},{"type": "indicator"}]])

    fig.add_trace(go.Indicator(
            mode = 'number', 
            value = int(hittil),
            number = {'valueformat': ',', 'prefix' :'$'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Kostnad hittil i år (eksludert inneværende uke)")}
        ), row = 1, col = 1)
    fig.add_trace(go.Indicator(
            mode = 'number', 
            value = int(resten),
            number = {'valueformat': ',', 'prefix' :'$'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Forventet kostnad resten av året")}
        ), row = 2, col = 1)
    fig.add_trace(go.Indicator(
            mode = 'number', 
            value = int(totalt),
            number = {'valueformat': ',', 'prefix' :'$'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Forventet totalkostnad i år")}
        ), row = 1, col = 2)
    fig.add_trace(go.Indicator(
            mode = 'number', 
            value = int(ifjor),
            number = {'valueformat': ',', 'prefix' :'$'},
            title = {'text': '{0}<br><span style=\'font-size:0.7em;color:gray\'>{1}</span>'.format("","Totalkostnad i fjor")}
        ), row = 2, col = 2)
    return fig


def prepare_change_dataframes(df_days):
    # Ser ikke på gårsdagen fordi den ikke er ferdig oppdatert på kjøretidspunktet
    df_previous_7_days = df_days[df_days.dato >= (datetime.today() - timedelta(8)).date()]
    df_earlier = df_days[df_days.dato < (datetime.today() - timedelta(8)).date()]
    df_previous_grouped = df_previous_7_days.groupby(["service_description", "env"], as_index=False).calculated_cost.mean()
    df_earlier_grouped = df_earlier.groupby(["service_description", "env"], as_index=False).calculated_cost.mean()  
    
    df_change = df_previous_grouped.merge(df_earlier_grouped,
                              on=["service_description", "env"],
                              suffixes=["", "_mean"])
    df_change["growth_euro"] = df_change.calculated_cost - df_change.calculated_cost_mean
    df_change["growth_percent"] = df_change.growth_euro / df_change.calculated_cost_mean
    
    df_highest_percent = df_change.sort_values("growth_percent", ascending=False).reset_index(drop=True)
    df_highest_percent = df_highest_percent[df_highest_percent.calculated_cost > 10]
    df_highest_euro = df_change.sort_values("growth_euro", ascending=False).reset_index(drop=True)
    return df_highest_percent, df_highest_euro


def make_change_figs(df_highest_percent, df_highest_euro) -> dict:
    labels = {"service_description":"Produkt", "growth_percent":"Endring (%)", "growth_euro":"Endring (€)"}
    change_figs = {}

    fig_percent = px.bar(df_highest_percent[df_highest_percent.env == 'prod'], "service_description", "growth_percent", height=600, hover_data=["service_description", "growth_euro", "growth_percent"], labels=labels, title="Kostnadsendring i prod (%)")
    fig_percent.update_layout(yaxis={'tickformat':".0%"})
    fig_euro = px.bar(df_highest_euro[df_highest_euro.env == 'prod'], "service_description", "growth_euro", height=600, custom_data=["growth_percent"], labels=labels, title="Kostnadsendring i prod (€)")
    fig_euro.update_traces(hovertemplate="<br>".join([
            "%{x}",
            "Endring (€): %{y}",
            "Endring (%): %{customdata[0]: .0%}"
        ]))
    change_figs["percent_prod"] = fig_percent
    change_figs["euro_prod"] = fig_euro

    fig_percent_dev = px.bar(df_highest_percent[df_highest_percent.env == 'dev'], "service_description", "growth_percent", height=600, hover_data=["service_description", "growth_euro", "growth_percent"], labels=labels, title="Kostnadsendring i dev (%)")
    fig_percent_dev.update_layout(yaxis={'tickformat':".0%"})
    fig_euro_dev = px.bar(df_highest_euro[df_highest_euro.env == 'dev'], "service_description", "growth_euro", height=600, custom_data=["growth_percent"], labels=labels, title="Kostnadsendring i dev (€)")
    fig_euro_dev.update_traces(hovertemplate="<br>".join([
            "%{x}",
            "Endring (€): %{y}",
            "Endring (%): %{customdata[0]: .0%}"
        ]))
    change_figs["percent_dev"] = fig_percent_dev
    change_figs["euro_dev"] = fig_euro_dev

    # fig_percent_rest = px.bar(df_highest_percent[~df_highest_percent.env.isin(["dev", "prod"])], "service_description", "growth_percent", height=600, hover_data=["service_description", "growth_euro", "growth_percent", "env"], labels=labels, title="Kostnadsendring på tjenester uten prod/dev-label (%)")
    # fig_percent_rest.update_layout(yaxis={'tickformat':".0%"})
    # fig_euro_rest = px.bar(df_highest_euro[~df_highest_euro.env.isin(["dev", "prod"])], "service_description", "growth_euro", height=600, custom_data=["growth_percent", "env"], labels=labels, title="Kostnadsendring på tjenester uten prod/dev-label (€)")
    # fig_euro_rest.update_traces(hovertemplate="<br>".join([
    #         "%{x}",
    #         "Endring (€): %{y}",
    #         "Endring (%): %{customdata[0]: .0%}",
    #         "Env: %{customdata[1]}" 
    #     ]))
    # change_figs["percent_rest"] = fig_percent_rest
    # change_figs["euro_rest"] = fig_euro_rest

    return change_figs


def prepare_datastory(figs, weeks_to_look_at, months_to_look_at):
    ds = DataStory("Kostnader i GCP og Aiven")

    ds.header("Kostnader i GCP")
    ds.markdown("Det er strammere tider og derfor større behov for å ha kontroll på store utgiftsposter. Her har vi prøvd å predikere hvor mye penger Nav kommer til å bruke på GCP og Aiven i 2023.")
    ds.header("Nøkkeltall", level=3)
    ds.markdown(f"""Forventninger er basert på utvikling siste {weeks_to_look_at} uker. 
                Merk at det er vanskelig å spå framtiden, og forventningene er basert på en lite avansert modell. 
                Derfor er det viktig å være klar over at forventede kostnader sannsynligvis ikke treffer presist på faktisk kostnad. 
                Modellen kan potensielt bomme med flere hundre tusen euro over et år.
                """)
    ds.markdown("""NB! Et år er her definert fra 1. november til 31. oktober for å enklere kunne sammenligne med regnskap.""")
    ds.plotly(figs["numbers_gcp"].to_json())

    ds.header("Kostnader per service siden 1.november 2021.", level=3)
    ds.markdown("Merk at det mangler noe data fra uke 2 i 2022, slik at totalen denne uka ser lavere ut enn den var.")
    ds.plotly(figs["services_gcp"].to_json())

    ds.header(f"Forventet utvikling i kostnader ut oktober 2023 basert på utvikling siste {weeks_to_look_at} uker", level=3)
    ds.markdown("1.november 2021 var mandag uke 44. Fem av dagene i uke 44 2023 er i november, og regnes derfor ikke med.")
    ds.plotly(figs["forecast_gcp"].to_json())

    ds.header(f"Historisk prediksjon", level=3)
    ds.markdown(f"""Figuren viser hva den forventede totalkostnaden for budsjettåret 2023 var på ulike tidspunkt.
                Formålet er å få et bilde på hvor stor usikkerhet det var i prediksjonen på starten av året og hvordan den blir mindre etter som vi nærmer oss slutten av året.  
                X-aksen viser hvor mange uker det er siden hver prediksjon ble gjort.
                Det vil si at 0 angir siste prediksjon av totalkostnad for 2023 og første prediksjon ble gjort i starten av oktober 2022.""")
    ds.plotly(figs["historic_gcp"].to_json())

    ds.header(f"Kostnader i Aiven")
    ds.markdown("Vi gjentar samme prosess for Aivenkostnader. Merk at vi her kun har månedlige data og at valutaen er amerikanske dollar.")
    ds.header("Nøkkeltall", level=3)
    ds.plotly(figs["numbers_aiven"].to_json())
    ds.header("Kostnader per service siden 1.november 2021.", level=3)
    ds.plotly(figs["services_aiven"].to_json())
    ds.header(f"Forventet utvikling i kostnader ut oktober 2023 basert på utvikling siste {months_to_look_at} måneder", level=3)
    ds.plotly(figs["forecast_aiven"].to_json())


    ds.header("Største endringer i kostnad i prod-gcp.", level=2)
    ds.markdown("For å ikke få store overraskelser må vi følge med på store endringer i kostnad for de ulike tjenestene. ")
    if datetime.now().month == 7:
        ds.header("NB! Det ble gjort en større endring i labels for prod og dev i slutten av juni. Dette får noen rare utslag i de følgende figurene fram til det har gått tre uker siden endringen.", level=3)
    ds.markdown("Vi sammenligner siste 7 dager (til og med to dager siden) med to foregående uker. Eksempel: Hvis datoen i dag var 09.02. ville vi sett på perioden 01.02.-07.02. (7 dager) sammenlignet med perioden 18.01.-31.01. (14 dager).")
    ds.plotly(figs["euro_prod"].to_json())
    ds.markdown("For prosentvis endring ser vi kun på produkter som i snitt koster minst 5€ per dag.")
    ds.plotly(figs["percent_prod"].to_json())
    ds.header("Største endringer i kostnad i dev-gcp. Siste 7 dager sammenlignet med to foregående uker.", level=3)
    ds.plotly(figs["euro_dev"].to_json())
    ds.markdown("For prosentvis endring ser vi kun på produkter som i snitt koster minst 5€ per dag.")
    ds.plotly(figs["percent_dev"].to_json())
    return ds


if __name__=='__main__':
    FIRST_YEAR = 2021 # 1. november 2021
    FIRST_WEEK = 44 # 1. november 2021
    FIRST_MONTH = 11 # 1. november 2021
    FIRST_FIN_YEAR = 2022 # Budget year nov-oct
    CURRENT_YEAR = datetime.now().year
    WEEKS_TO_LOOK_AT = 15 # Number of weeks in training set
    MONTHS_TO_LOOK_AT = 10 # Number of months to look at for Aivendata
    figs = {}

    Client = bigquery.Client('nais-analyse-prod-2dcc')
    
    # GCP: 
    df_bq = load_data_from_bq(Client, query_number=1)

    df_service, current_n_week = prepare_df_service(df_bq)
    figs["services_gcp"] = make_services_fig(df_service)

    x_s, y_s = make_training_data(df_service, current_n_week, WEEKS_TO_LOOK_AT, n_top=2)
    mod = LinearRegression().fit(x_s.values, y_s.values)

    previous_n_week, n_weeks, n_services = make_week_vars(current_n_week, x_s)
    pred = predict_cost(mod, current_n_week, n_weeks, n_services)
    df_pred = pred_to_df(pred, x_s, current_n_week)
    kostnad_i_fjor, kostnad_hittil, forventet_resten, forventet_totalt = compute_total_costs(df_bq, df_pred, current_n_week)
    print(f"Kostnad i fjor: {int(kostnad_i_fjor)} € \nKostnad hittil i år: {int(kostnad_hittil)} € \nForventet gjenstående kostnad i år: {int(forventet_resten)} € \nForventet totalkostnad i år: {int(forventet_totalt)} €")
    figs["numbers_gcp"] = make_numbers_fig(kostnad_i_fjor, kostnad_hittil, forventet_resten, forventet_totalt)
    figs["forecast_gcp"] = make_forecast_fig(df_service, df_pred, current_n_week)
    figs["historic_gcp"] = prediction_history(current_n_week, df_service, WEEKS_TO_LOOK_AT)

    # Aiven:

    df_aiven = load_data_from_bq(Client, query_number=2)
    figs["services_aiven"] = make_aiven_services_fig(df_aiven)
    df_aiven_month = group_aiven_month(df_aiven)
    current_n_month, n_months = make_month_vars(df_aiven_month)
    
    x_a, y_a = make_training_data_aiven(df_aiven_month, current_n_month, MONTHS_TO_LOOK_AT)
    mod_a = linear_regression_aiven(x_a, y_a)
    
    df_pred_aiven = predict_aiven(mod_a, n_months, current_n_month)
    figs["forecast_aiven"] = make_forecast_fig_aiven(df_aiven_month, df_pred_aiven)
    a_kostnad_i_fjor, a_kostnad_hittil, a_forventet_resten, a_forventet_totalt = compute_total_costs_aiven(df_aiven, df_pred_aiven)
    figs["numbers_aiven"] = make_numbers_fig_aiven(a_kostnad_i_fjor, a_kostnad_hittil, a_forventet_resten, a_forventet_totalt)

    # Cost changes:

    df_days = load_data_from_bq(Client, query_number=3)
    df_highest_percent, df_highest_euro = prepare_change_dataframes(df_days)
    figs.update(make_change_figs(df_highest_percent, df_highest_euro))

    story = prepare_datastory(figs, WEEKS_TO_LOOK_AT, MONTHS_TO_LOOK_AT)
    story.update(url="https://nada.intern.nav.no/api", token=os.environ["KOSTNAD_STORY_TOKEN"])
