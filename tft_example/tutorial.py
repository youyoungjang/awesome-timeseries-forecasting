import numpy as np
import pandas as pd
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from data_utils.utils import get_stock_price


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)


# today = datetime.strftime(datetime.today(), '%Y-%m-%d')
# df = get_stock_price(ticker='AXP', start_date='2000-01-01', end_date=today)
#
# df = (
#     df.reset_index()
#     .rename({'index': 'time_idx'}, axis=1)
# )
#
# df['group'] = 0

# df = pd.melt(
#     df,
#     id_vars=['time_idx', 'date'],
#     value_vars=['price', 'volume'],
#     var_name='group',
#     value_name='value',
# )

from pytorch_forecasting.data.examples import get_stallion_data

df = get_stallion_data()

# add time index
df["time_idx"] = df["date"].dt.year * 12 + df["date"].dt.month
df["time_idx"] -= df["time_idx"].min()

# add additional features
df["month"] = df.date.dt.month.astype(str).astype("category")  # categories have be strings
df["log_volume"] = np.log(df.volume + 1e-8)
df["avg_volume_by_sku"] = df.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
df["avg_volume_by_agency"] = df.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
df[special_days] = df[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")

# define dataset
max_prediction_length = 6
max_encoder_length = 24
batch_size = 128
training_cutoff = df['time_idx'].max() - max_prediction_length

train_dataset = TimeSeriesDataSet(
    data=df[lambda x: x.time_idx <= training_cutoff],
    time_idx='time_idx',
    target='volume',
    group_ids=['agency', 'sku'],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    variable_groups={"special_days": special_days},
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df,
    predict=True,
    # min_prediction_idx=train_dataset.index.time.max()+1,
    stop_randomization=True,
)

# create dataloaders
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 10)

# test baseline model
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print((actuals - baseline_predictions).abs().mean().item())


# define trainer
callbacks = [
    ModelCheckpoint(
        dirpath='files',
        filename='tft',
        monitor='val_loss',
        verbose=True,
        save_weights_only=True,
        mode='min',
        every_n_epochs=1,
    ),
    LearningRateMonitor(),
    EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode='min',
    )
]

logger = TensorBoardLogger('lightning_logs')

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    num_nodes=1,
    devices=1,
    max_epochs=30,
    accelerator='gpu',
    num_sanity_val_steps=2,
    deterministic=False,
    benchmark=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    enable_model_summary=True,
    enable_progress_bar=True,
)

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)
