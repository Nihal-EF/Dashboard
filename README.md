# Dashboard

## Commande pour déployer l'API
mlflow models serve --env-manager=local --host=0.0.0.0 -m mlflow_model_pyfunc/

## Commande pour appeler l'API
curl http://35.180.69.207:5000/invocations -H "Content-Type:application/json" -d '{"dataframe_split":{"columns":["EXT_SOURCE_1","EXT_SOURCE_2","DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "PAYMENT_RATE", "ANNUITY_INCOME_PERC", "DPD_BOOL"],"data":[[1, 2, 3, 4, 5, 6, 7, 8]]}}'
