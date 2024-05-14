# nais-cloud-cost-prediction

# Replaced by https://github.com/navikt/nais-cloud-cost-prediction-quarto

Daily prediction of future costs in GCP and Aiven. 

Needs read access to the BigQuery table `nais-analyse-prod-2dcc.nais_billing_nav.cost_breakdown_gcp` through service account credentials `SA_KEY` and the datastory's update token `KOSTNAD_STORY_TOKEN`.

To run locally you need personal access to the table (log in with `gcloud auth login --update-adc`) and to be part of the group `nais-analyse`
