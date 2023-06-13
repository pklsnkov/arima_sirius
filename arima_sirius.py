import pandas as pd
import scipy.stats as stats
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

train_path = Path().cwd().parent / 'data' / 'train.csv'
debit = pd.read_csv(train_path, parse_dates=['datetime'])
oil_wells = debit.groupby('Номер скважины')
all_results_df = pd.DataFrame()


def forecast_oil_well(oil_well):
    """
    The function uses the integrated ARIMA autoregression model to predict the model.
    The parameters inside the model are determined using the maximum likelihood function."
    """

    debit = oil_well["Дебит нефти"].values
    if stats.normaltest(debit).pvalue > 0.05:
        p = 0
    else:
        p = 1
    try:
        model = ARIMA(debit, order=(p, 1, 1))
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=90)
    except:
        model = ARIMA(debit, order=(1, 1, 1))
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=90)
    results_df = pd.DataFrame({'datetime': pd.date_range(oil_well['datetime'].iloc[-1], periods=90, freq='1D'),
                               'Номер скважины': oil_well['Номер скважины'].iloc[0],
                               'forecast': forecast})
    return results_df


for oil_well_num, oil_well in oil_wells:
    oil_well_results_df = forecast_oil_well(oil_well)
    all_results_df = pd.concat([all_results_df, oil_well_results_df], ignore_index=True)

all_results_df.to_csv('forecast.csv', index=False)
