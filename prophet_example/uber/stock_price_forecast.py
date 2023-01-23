import matplotlib.pyplot as plt
from datetime import datetime
from data_utils.utils import get_stock_price_for_prophet

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


today = datetime.strftime(datetime.today(), '%Y-%m-%d')
df = get_stock_price_for_prophet('UBER', '2000-01-01', today)

# split
test_days = 30
point = int(df.shape[0] - test_days)
train = df[:point]
test = df[point:]

prophet = Prophet(
    daily_seasonality=False,
    growth='linear',
    yearly_seasonality='auto',
    weekly_seasonality='auto',
    holidays=None,
    seasonality_mode='additive',
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    changepoint_prior_scale=0.05,
)

prophet.fit(train)

future = prophet.make_future_dataframe(periods=test_days)
forecast = prophet.predict(future)
prophet.plot(forecast)
# plt.savefig('uber.png', dpi=300)
plt.show()

prophet.plot_components(forecast)
plt.show()

# built in fbprophet evaluation function
# initial: first train period
# period: how many days to add in the next train process
# horizon: evaluation period
evaluation_model = cross_validation(prophet, initial='150 days', period='7 days', horizon='30 days')
evaluation_metrics = performance_metrics(evaluation_model)
print(evaluation_metrics[['horizon', 'mse', 'rmse', 'mae', 'coverage']])

