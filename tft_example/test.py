import numpy as np
import pandas as pd
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss

from data_utils.utils import get_stock_price


today = datetime.strftime(datetime.today(), '%Y-%m-%d')
df = get_stock_price(ticker='AMD', start_date='2000-01-01', end_date=today)

df = (
    df.reset_index()
    .rename({'index': 'time_idx'}, axis=1)
)

df['group'] = 0

df = pd.melt(
    df,
    id_vars=['time_idx', 'date'],
    value_vars=['price', 'volume'],
    var_name='group',
    value_name='value',
)


