import pandas as pd
import yfinance as yf
from datetime import datetime


def get_stock_price(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # info = yf.Ticker(ticker).info
    price = (
        yf.download(ticker, start_date, end_date)
        .reset_index()
        # .rename({'Date': 'date', 'Adj Close': 'price', 'Volume': 'volume'}, axis=1)
        .rename({'Date': 'ds', 'Adj Close': 'y'}, axis=1)
    )
    # price = price[['date', 'price', 'volume']]
    # price['date'] = price['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    price = price[['ds', 'y']]
    price['ds'] = price['ds'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

    return price
