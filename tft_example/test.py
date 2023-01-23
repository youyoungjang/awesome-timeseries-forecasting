import pandas as pd
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics.quantile import QuantileLoss

from data_utils.utils import get_stock_price


today = datetime.strftime(datetime.today(), '%Y-%m-%d')
df = get_stock_price(ticker='AMD', start_date='2000-01-01', end_date=today)

df = (
    df.reset_index()
    .rename({'index': 'time_idx'}, axis=1)
)

df = pd.melt(
    df,
    id_vars=['time_idx', 'date'],
    value_vars=['price', 'volume'],
    var_name='group',
    value_name='value',
)

batch_size = 128
test_start_date = '2022-07-01'

train_dataset = TimeSeriesDataSet(
    data=df[lambda x: x.date < test_start_date],
    group_ids=['group'],
    target='value',
    time_idx='time_idx',
    min_encoder_length=30,
    max_encoder_length=180,
    min_prediction_idx=365,
    max_prediction_length=30,
    time_varying_unknown_reals=['value'],
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df,
    min_prediction_idx=train_dataset.index.time.max()+1,
    stop_randomization=True,
)

# create dataloaders
train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size)

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
        patience=2,
        verbose=True,
        mode='min',
    )
]

logger = WandbLogger(
    project='temporal_fusion_transformer',
    name='test'
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    num_nodes=1,
    devices=1,
    max_epochs=10,
    accelerator='gpu',
    num_sanity_val_steps=2,
    deterministic=False,
    benchmark=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,
)

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4
)

trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)

